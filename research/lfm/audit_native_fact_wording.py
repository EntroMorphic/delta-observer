#!/usr/bin/env python3
"""Cross operand triples and task wording without retraining or supplying answers."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.audit_native_rules import extract_answer
from research.lfm.evaluate_pretrained_development import chat_inputs
from research.lfm.native_lora import install_lora, load_adapter
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.probe_lfm import choose_device


def render(a,b,c,wording):
    if wording == 'training':
        return (f'A delivery contains {a} trays holding {b} cups each. Then {c} cups are removed '
                'from the delivery. How many remain? Return only the integer.')
    if wording == 'diagnostic':
        return (f'An order consists of {a} racks with {b} plates per rack. {c} plates are taken '
                'out of the order. State only the integer number of plates left.')
    raise ValueError('Unknown wording')


def crossing(training, diagnostic):
    compound=[r for r in training if r.get('mode')=='compound_final']
    tasks=[r for r in diagnostic['tasks'] if r['category']=='arithmetic']
    if len(compound)!=128 or len(tasks)!=20:
        raise ValueError('Unexpected source counts')
    def parse(row,wording):
        pattern=(r'A delivery contains (\d+) trays holding (\d+) cups each\. Then (\d+) cups are removed from the delivery\. How many remain\? Return only the integer\.'
            if wording=='training' else
            r'An order consists of (\d+) racks with (\d+) plates per rack\. (\d+) plates are taken out of the order\. State only the integer number of plates left\.')
        match=re.fullmatch(pattern,row['prompt'])
        if match is None:raise ValueError('Source wording differs')
        a,b,c=map(int,match.groups())
        if row.get('response',row.get('expected')) != str(a*b-c):
            raise ValueError('Source label differs from operation')
        if render(a,b,c,wording)!=row['prompt']:raise ValueError('Nonidentical rendering')
        return a,b,c
    seen=[parse(row,'training') for row in compound]
    other=[parse(row,'diagnostic') for row in tasks]
    result=[]
    for fact_source,triples in [('training',seen[:20]),('diagnostic',other)]:
        for index,(a,b,c) in enumerate(triples):
            for wording in ('training','diagnostic'):
                result.append({'id':f'{fact_source}-{index:02d}-{wording}',
                    'fact_source':fact_source,'wording':wording,'case':index,
                    'operands':[a,b,c],'prompt':render(a,b,c,wording),
                    'expected':str(a*b-c)})
    return result, sorted(set(seen)&set(other))


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    training_bytes=Path(args.data).read_bytes()
    test_bytes=Path(args.tasks).read_bytes()
    probes,overlap=crossing(json.loads(training_bytes),json.loads(test_bytes))
    (output/'probes.json').write_text(json.dumps(probes,indent=2))
    device=choose_device(args.device)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16).to(device).eval()
    config=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=192,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
    result={'protocol':'research/lfm/manifold/native_transfer_synth.md',
        'scope':'crossed exposed diagnostic, not sealed evaluation',
        'source_data_sha256':hashlib.sha256(training_bytes).hexdigest(),
        'source_tasks_sha256':hashlib.sha256(test_bytes).hexdigest(),
        'diagnostic_triples_already_in_training':overlap,
        'generation_config':config.to_dict(),'arms':{}}
    for arm in ('baseline','rank8','rank32'):
        provenance=None
        if arm!='baseline':
            path=Path(args.checkpoint_root)/f'native-fit-ladder-{arm}/checkpoint.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            c=checkpoint['config']
            if c['model']!=MODEL or c['revision']!=REVISION:raise ValueError('Wrong base')
            # Rank differs: reload pristine base before installing each adapter.
            if arm=='rank32':
                model.cpu();del model
                model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16).to(device).eval()
            targets=install_lora(model,c['rank'],c['alpha'])
            if targets!=c['targets']:raise ValueError('Wrong adapter targets')
            load_adapter(model,checkpoint['adapter'])
            provenance={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        rows=[]
        for probe in probes:
            inputs=chat_inputs(tokenizer,probe['prompt']).to(device)
            with torch.inference_mode():
                ids=model.generate(**inputs,generation_config=config)[0,inputs['input_ids'].shape[1]:].tolist()
            text=tokenizer.decode(ids,skip_special_tokens=True)
            parsed=extract_answer({'family':'arithmetic'},text)
            rows.append({**probe,'response':text,'completion_ids':ids,
                'strict':text.strip()==probe['expected'],
                'parsed_content':parsed==probe['expected'] if parsed is not None else None,
                'ended_with_eos':bool(ids and ids[-1]==tokenizer.eos_token_id)})
        cells={}
        for facts in ('training','diagnostic'):
            for wording in ('training','diagnostic'):
                selected=[r for r in rows if r['fact_source']==facts and r['wording']==wording]
                cells[f'{facts}_facts_{wording}_wording']={'n':len(selected),
                    'strict':sum(r['strict'] for r in selected),
                    'parsed_content':sum(r['parsed_content'] is True for r in selected),
                    'unresolved_content':sum(r['parsed_content'] is None for r in selected)}
        result['arms'][arm]={'checkpoint':provenance,'rows':rows,'cells':cells}
        print(arm,json.dumps(cells),flush=True)
        (output/'crossing.json').write_text(json.dumps(result,indent=2))
    result['complete']=True
    (output/'crossing.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','tasks','checkpoint-root','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
