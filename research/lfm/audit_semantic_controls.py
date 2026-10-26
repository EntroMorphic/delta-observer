#!/usr/bin/env python3
"""Equal-length operator controls and native-retention token failure traces."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

from research.lfm.audit_semantic_atomics import capture,patched_forward,distribution_metrics,digest
from research.lfm.evaluate_pretrained_development import chat_inputs,content_validation
from research.lfm.native_lora import install_lora,load_adapter
from research.lfm.native_lora_smoke import MODEL,REVISION
from research.lfm.probe_lfm import choose_device


def balanced_prompt(fact,operation):
    a,b,c=fact
    word='loses' if operation=='subtract' else 'gains'
    return (f'A warehouse stores {a} crates with {b} cups in each crate. It subsequently {word} {c} cups. '
            'State the number of cups now in storage. Reply only with the decimal integer.')


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    data=Path(args.data);tasks=json.loads((data/'tasks.json').read_text())['tasks']
    pairs=[t for t in tasks if t['split']=='new_facts_new_template']
    retention=[t for t in tasks if t['category']=='json_record']
    baseline_path=Path(args.evaluations)/'native_semantic_transfer_v1_baseline/baseline.json'
    baseline={r['id']:r for r in json.loads(baseline_path.read_text())['results']}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    device=choose_device(args.device)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16).to(device).eval()
    config=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=192,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
    result={'scope':'equal-token-length one-word intervention and retention diagnostics; no training',
        'tasks_sha256':digest(data/'tasks.json'),'baseline_evaluation_sha256':digest(baseline_path),
        'arms':{}}
    for arm in ('baseline','repeat','paraphrase','contrast'):
        if arm!='baseline':
            path=Path(args.checkpoint_root)/f'native-semantic-transfer-v1-{arm}/checkpoint.pt'
            c=torch.load(path,map_location='cpu',weights_only=True)
            if c['config']['revision']!=REVISION:raise ValueError('Revision mismatch')
            if arm=='repeat':install_lora(model,c['config']['rank'],c['config']['alpha'])
            load_adapter(model,c['adapter'])
        record={'balanced_pairs':[],'retention':[]}
        evaluation=json.loads((Path(args.evaluations)/f'native_semantic_transfer_v1_{arm}/baseline.json').read_text())
        generated={r['id']:r for r in evaluation['results']}
        for task in pairs:
            fact=task['operands'];a,b,c=fact
            inputs=[chat_inputs(tokenizer,balanced_prompt(fact,op)).to(device) for op in ('subtract','add')]
            ids=[x['input_ids'][0].tolist() for x in inputs]
            if len(ids[0])!=len(ids[1]) or sum(a!=b for a,b in zip(*ids))!=1:
                raise ValueError('Operator control must differ in exactly one token')
            minus=tokenizer.encode(str(a*b-c),add_special_tokens=False)
            plus=tokenizer.encode(str(a*b+c),add_special_tokens=False)
            if len(minus)!=1 or len(plus)!=1:raise ValueError('Single-token answer required')
            minus,plus=minus[0],plus[0]
            captured=[capture(model,x) for x in inputs]
            metrics=[distribution_metrics(logits,t,o) for (logits,_),t,o in zip(captured,(minus,plus),(plus,minus))]
            patches=[]
            for depth in (2,5,8,10,12,13,14,15):
                logits=patched_forward(model,inputs[0],depth,captured[1][1][f'{depth:02d}_after_ffn'])
                patches.append({'layer':depth,'metrics':distribution_metrics(logits,minus,plus)})
            record['balanced_pairs'].append({'id':task['id'],'operands':fact,'input_ids':ids,
                'subtract':metrics[0],'add':metrics[1],
                'selectivity_shift':metrics[0]['target_minus_alternative_logit']+metrics[1]['target_minus_alternative_logit'],
                'patches':patches})
        for task in retention:
            # Probe the baseline's full generated continuation as a fixed history.
            reference=baseline[task['id']]['generated_token_ids']
            inputs=chat_inputs(tokenizer,task['prompt']).to(device)
            plen=inputs['input_ids'].shape[1]
            full=torch.cat((inputs['input_ids'],torch.tensor([reference],device=device)),dim=1)
            with torch.inference_mode():
                logits=model(full[:,:-1],use_cache=False).logits[0,plen-1:].float()
                lp=logits.log_softmax(-1)
            rows=[]
            for offset,target in enumerate(reference):
                target=int(target);pred=int(logits[offset].argmax())
                rows.append({'offset':offset,'target_id':target,'target_piece':tokenizer.decode([target]),
                    'predicted_id':pred,'predicted_piece':tokenizer.decode([pred]),
                    'target_probability':float(lp[offset,target].exp()),
                    'eos_probability':float(lp[offset,tokenizer.eos_token_id].exp()),
                    'target_vs_top1_margin':float(logits[offset,target]-logits[offset].max())})
            # Delay EOS only through eight generated tokens, then allow normal
            # stopping. Suppressing it forever would manufacture trailing junk.
            with torch.inference_mode():
                ids=model.generate(**inputs,generation_config=config,
                    min_new_tokens=8)[0,plen:].tolist()
            suppressed=tokenizer.decode(ids,skip_special_tokens=True)
            actual=generated[task['id']]['generated_token_ids']
            difference=next((i for i,(a,b) in enumerate(zip(actual,reference)) if a!=b),None)
            if difference is None and len(actual)!=len(reference):difference=min(len(actual),len(reference))
            record['retention'].append({'id':task['id'],'native_reference_text':baseline[task['id']]['response'],
                'actual_response':generated[task['id']]['response'],'first_generated_difference':difference,
                'native_reference_token_metrics':rows,'eos_delayed_until_token':8,
                'eos_delayed_response':suppressed,
                'eos_delayed_content':content_validation(task,suppressed)['pass'],
                'delay_hit_budget':len(ids)==192})
        result['arms'][arm]=record
        (output/'controls.json').write_text(json.dumps(result,indent=2))
        print(arm,'balanced top1 both',sum(r['subtract']['target_rank']==1 and r['add']['target_rank']==1 for r in record['balanced_pairs']),flush=True)
    result['complete']=True
    (output/'controls.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint-root','evaluations','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
