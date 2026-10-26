#!/usr/bin/env python3
"""Measured answer exposure, operator margins, residual routing, and state swaps."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

from research.lfm.evaluate_pretrained_development import chat_inputs
from research.lfm.native_lora import install_lora,load_adapter
from research.lfm.native_lora_smoke import MODEL,REVISION
from research.lfm.probe_lfm import choose_device


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution_metrics(logits,target,alternative):
    logits=logits.float();lp=logits.log_softmax(-1);p=lp.exp()
    top=logits.topk(5)
    return {'target_id':target,'alternative_id':alternative,'top1_id':int(logits.argmax()),
        'target_rank':int(1+(logits>logits[target]).sum()),
        'target_probability':float(p[target]),'alternative_probability':float(p[alternative]),
        'target_minus_alternative_logit':float(logits[target]-logits[alternative]),
        'target_minus_top1_logit':float(logits[target]-logits.max()),
        'entropy_nats':float(-(p*lp).sum()),'top64_mass':float(p.topk(64).values.sum()),
        'top5_ids':top.indices.tolist(),'top5_logits':top.values.tolist()}


def capture(model,inputs):
    states={};handles=[]
    for depth,layer in enumerate(model.model.layers):
        def pre(_module,args,d=depth):states[f'{d:02d}_before_operator']=args[0][0,-1].detach().clone()
        def after_operator(_module,args,d=depth):states[f'{d:02d}_after_operator']=args[0][0,-1].detach().clone()
        def post(_module,args,output,d=depth):states[f'{d:02d}_after_ffn']=output[0,-1].detach().clone()
        handles.extend((layer.register_forward_pre_hook(pre),
                        layer.ffn_norm.register_forward_pre_hook(after_operator),
                        layer.register_forward_hook(post)))
    try:
        with torch.inference_mode():logits=model(**inputs,use_cache=False).logits[0,-1].detach()
    finally:
        for handle in handles:handle.remove()
    return logits,states


def patched_forward(model,inputs,depth,donor):
    def hook(_module,_args,output):
        patched=output.clone();patched[0,-1]=donor
        return patched
    handle=model.model.layers[depth].register_forward_hook(hook)
    try:
        with torch.inference_mode():return model(**inputs,use_cache=False).logits[0,-1].detach()
    finally:handle.remove()


def exposure(data,evaluation,tasks,tokenizer):
    groups=json.loads(data.read_text());rows=[r for g in groups for r in g['rows']]
    frequencies=Counter(r['response'] for r in rows)
    supervised=Counter(i for r in rows for i in r['labels'] if i!=-100)
    evaluated={r['id']:r for r in evaluation['results']}
    buckets={}
    for split in sorted({t['split'] for t in tasks if t['category']=='arithmetic'}):
        for covered in (False,True):
            selected=[t for t in tasks if t['split']==split and (t['expected'] in frequencies)==covered]
            buckets[f'{split}_answer_seen_{covered}']={'n':len(selected),
                'strict_successes':sum(evaluated[t['id']]['validation']['pass'] is True for t in selected)}
    return {'training_file_sha256':digest(data),'sequence_exposures':len(rows),
        'unique_prompt_strings':len({r['prompt'] for r in rows}),
        'distinct_answer_strings':len(frequencies),'answer_frequency':dict(frequencies),
        'supervised_label_token_counts':{str(k):v for k,v in supervised.items()},
        'eos_supervised_fraction':supervised[tokenizer.eos_token_id]/sum(supervised.values()),
        'coverage_buckets':buckets,
        'task_answer_exposures':{t['id']:frequencies[t['expected']] for t in tasks if t['category']=='arithmetic'}}


def run(args):
    root=Path(args.checkpoint_root);data=Path(args.data);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    tasks=json.loads((data/'tasks.json').read_text())['tasks'];by_id={t['id']:t for t in tasks}
    manifest=json.loads((data/'manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    device=choose_device(args.device)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16).to(device).eval()
    result={'scope':'exposed diagnostic; no training; conditional first-answer-token and residual interventions',
        'model':MODEL,'revision':REVISION,'dtype':'float16','device':device,
        'tasks_sha256':digest(data/'tasks.json'),'arms':{},
        'patch_caveat':'Only the final prompt-token residual is swapped. Donor states can contain all context features; this is not unique attribution to the operation word. Sequence lengths may differ.',
        'logit_lens_caveat':'Intermediate probes use final normalization and frozen output head; they are not the actual final decision at that depth.'}
    for arm in ('baseline','repeat','paraphrase','contrast'):
        checkpoint_hash=None
        if arm!='baseline':
            checkpoint_path=root/f'native-semantic-transfer-v1-{arm}/checkpoint.pt'
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            config=checkpoint['config']
            if config['revision']!=REVISION or config['model']!=MODEL:raise ValueError('Base mismatch')
            if arm=='repeat':install_lora(model,config['rank'],config['alpha'])
            load_adapter(model,checkpoint['adapter']);checkpoint_hash=digest(checkpoint_path)
        eval_path=Path(args.evaluations)/f'native_semantic_transfer_v1_{arm}/baseline.json'
        evaluated=json.loads(eval_path.read_text());generated={r['id']:r for r in evaluated['results']}
        if evaluated['tasks_sha256']!=result['tasks_sha256']:raise ValueError('Evaluation tasks differ')
        record={'checkpoint_sha256':checkpoint_hash,'evaluation_sha256':digest(eval_path),'pairs':[]}
        if arm!='baseline':
            source=data/manifest['arms'][arm]['path']
            if digest(source)!=manifest['arms'][arm]['sha256']:raise ValueError('Training source changed')
            record['exposure']=exposure(source,evaluated,tasks,tokenizer)
        for index in range(20):
            subtract=by_id[f'new_facts_new_template-{index:02d}'];add=by_id[f'counterfactual_add-{index:02d}']
            target_ids=[tokenizer.encode(t['expected'],add_special_tokens=False) for t in (subtract,add)]
            if any(len(ids)!=1 for ids in target_ids):raise ValueError('Requires a single answer token')
            minus,plus=[ids[0] for ids in target_ids]
            inputs=[chat_inputs(tokenizer,t['prompt']).to(device) for t in (subtract,add)]
            captured=[capture(model,x) for x in inputs]
            row={'case':index,'operands':subtract['operands'],'minus_token':minus,'plus_token':plus,
                'input_lengths':[x['input_ids'].shape[1] for x in inputs],
                'first_token_metrics':{},'stages':[],'patches':[]}
            for label,task,(logits,states),target,other in zip(('subtract','add'),(subtract,add),captured,(minus,plus),(plus,minus)):
                metrics=distribution_metrics(logits,target,other)
                metrics['top1_piece']=tokenizer.decode([metrics['top1_id']])
                metrics['top5_pieces']=[tokenizer.decode([i]) for i in metrics['top5_ids']]
                metrics['agrees_with_stored_generation_first_token']=metrics['top1_id']==generated[task['id']]['generated_token_ids'][0]
                row['first_token_metrics'][label]=metrics
            row['operation_selectivity_shift']=(float(captured[0][0][minus]-captured[0][0][plus])
                -float(captured[1][0][minus]-captured[1][0][plus]))
            with torch.inference_mode():
                for stage in captured[0][1]:
                    a,b=[entry[1][stage] for entry in captured]
                    lens=[]
                    for state in (a,b):
                        projected=model.lm_head(model.model.embedding_norm(state[None,None]))[0,0].float()
                        lens.append(float(projected[minus]-projected[plus]))
                    row['stages'].append({'stage':stage,'cosine':float(F.cosine_similarity(a.float()[None],b.float()[None])),
                        'state_difference_norm':float((a.float()-b.float()).norm()),
                        'state_norms':[float(a.float().norm()),float(b.float().norm())],
                        'lens_minus_vs_plus_margins':lens,'lens_operation_selectivity':lens[0]-lens[1]})
            # All 16 layers, both directions. Self-patch control at final layer.
            for depth in range(16):
                outcomes={}
                for source_index,label in ((0,'subtract_with_add_state'),(1,'add_with_subtract_state')):
                    donor=captured[1-source_index][1][f'{depth:02d}_after_ffn']
                    logits=patched_forward(model,inputs[source_index],depth,donor)
                    outcomes[label]=distribution_metrics(logits,
                        minus if source_index==0 else plus,plus if source_index==0 else minus)
                row['patches'].append({'layer':depth,**outcomes})
            self_patch=patched_forward(model,inputs[0],15,captured[0][1]['15_after_ffn'])
            row['self_patch_max_logit_error']=float((self_patch-captured[0][0]).abs().max())
            row['final_layer_cross_patch_matches_donor_top1']=[
                row['patches'][-1]['subtract_with_add_state']['top1_id']==int(captured[1][0].argmax()),
                row['patches'][-1]['add_with_subtract_state']['top1_id']==int(captured[0][0].argmax())]
            record['pairs'].append(row)
        result['arms'][arm]=record
        (output/'atomics.json').write_text(json.dumps(result,indent=2))
        print(arm,'pairs',len(record['pairs']),'self_patch_max',max(r['self_patch_max_logit_error'] for r in record['pairs']),flush=True)
    result['complete']=True
    (output/'atomics.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint-root','evaluations','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
