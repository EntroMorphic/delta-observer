#!/usr/bin/env python3
"""Same-prefix baseline activation patches and LoRA knockouts at schema failures."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

from research.lfm.audit_semantic_atomics import capture,patched_forward,distribution_metrics,digest
from research.lfm.evaluate_pretrained_development import chat_inputs,content_validation
from research.lfm.native_lora import LoRALinear,install_lora,load_adapter
from research.lfm.native_lora_smoke import MODEL,REVISION
from research.lfm.probe_lfm import choose_device


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    data=Path(args.data);tasks=json.loads((data/'tasks.json').read_text())['tasks']
    tasks=[t for t in tasks if t['id'] in ('record-0','record-1','record-4')]
    native=json.loads((Path(args.evaluations)/'native_semantic_transfer_v1_baseline/baseline.json').read_text())
    reference={r['id']:r for r in native['results']}
    device=choose_device(args.device);tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16).to(device).eval()
    donors={};prefixes={};targets={}
    for task in tasks:
        ids=reference[task['id']]['generated_token_ids']
        # Fixed first schema-key position: baseline prefix before its 'label' token.
        label=tokenizer.encode('label',add_special_tokens=False)
        if len(label)!=1 or ids[4]!=label[0]:raise ValueError('Unexpected native schema prefix')
        prompt=chat_inputs(tokenizer,task['prompt']).to(device)
        full=torch.cat((prompt['input_ids'],torch.tensor([ids[:4]],device=device)),dim=1)
        prefixes[task['id']]={'input_ids':full}
        targets[task['id']]=label[0]
        donors[task['id']]=capture(model,{'input_ids':full})
    generation=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=192,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
    result={'scope':'same prefix, target native schema key; baseline residual swaps and adapter knockouts',
        'tasks_sha256':digest(data/'tasks.json'),'arms':{},
        'interpretation':'Baseline-state swaps restore all represented features at that position, not a uniquely identified circuit. Knockouts change nonlinear interactions.'}
    for arm in ('repeat','paraphrase','contrast'):
        path=Path(args.checkpoint_root)/f'native-semantic-transfer-v1-{arm}/checkpoint.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['config']['revision']!=REVISION:raise ValueError('Revision mismatch')
        if arm=='repeat':install_lora(model,checkpoint['config']['rank'],checkpoint['config']['alpha'])
        load_adapter(model,checkpoint['adapter'])
        modules={name:module for name,module in model.named_modules() if isinstance(module,LoRALinear)}
        arm_record={'checkpoint_sha256':digest(path),'baseline_state_patches':[],'knockouts':[]}
        for task in tasks:
            inputs=prefixes[task['id']];target=targets[task['id']]
            actual,states=capture(model,inputs);competitor=int(actual.argmax())
            patched=[]
            for depth in range(16):
                logits=patched_forward(model,inputs,depth,donors[task['id']][1][f'{depth:02d}_after_ffn'])
                patched.append({'layer':depth,'metrics':distribution_metrics(logits,target,competitor)})
            arm_record['baseline_state_patches'].append({'id':task['id'],'prefix_ids':inputs['input_ids'][0].tolist(),
                'target_key_token':target,'full':distribution_metrics(actual,target,competitor),'patches':patched})
        interventions={'full':[],'all_off':list(modules),
            'attention_off':[name for name in modules if '.self_attn.' in name],
            'conv_off':[name for name in modules if '.conv.' in name]}
        interventions.update({f'layer_{depth:02d}_off':[name for name in modules if f'.layers.{depth}.' in name]
                             for depth in range(16)})
        for name,disabled in interventions.items():
            scales={key:module.scale for key,module in modules.items()}
            for key in disabled:modules[key].scale=0.
            rows=[]
            try:
                for task in tasks:
                    fixed=prefixes[task['id']]
                    with torch.inference_mode():logits=model(**fixed,use_cache=False).logits[0,-1]
                    metrics=distribution_metrics(logits,targets[task['id']],int(donors[task['id']][0].argmax()))
                    prompt=chat_inputs(tokenizer,task['prompt']).to(device)
                    with torch.inference_mode():
                        ids=model.generate(**prompt,generation_config=generation)[0,prompt['input_ids'].shape[1]:].tolist()
                    response=tokenizer.decode(ids,skip_special_tokens=True)
                    record={'id':task['id'],'key_metrics':metrics,'response':response,
                        'content':content_validation(task,response)['pass'],'completion_ids':ids}
                    if name=='all_off':record['native_logit_max_error']=float((logits-donors[task['id']][0]).abs().max())
                    rows.append(record)
            finally:
                for key,scale in scales.items():modules[key].scale=scale
            arm_record['knockouts'].append({'intervention':name,'disabled_modules':disabled,'cases':rows})
        result['arms'][arm]=arm_record
        (output/'schema.json').write_text(json.dumps(result,indent=2))
        print(arm,'complete',flush=True)
    result['complete']=True
    (output/'schema.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint-root','evaluations','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
