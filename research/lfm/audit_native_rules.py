#!/usr/bin/env python3
"""Controlled arithmetic decomposition and paired order-instruction probes."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.evaluate_pretrained_development import chat_inputs
from research.lfm.native_lora import install_lora, load_adapter
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.probe_lfm import choose_device


def probes():
    rows=[]
    for i in range(20):
        a,b,c=9+i,3+i%8,6+i
        modes={
            'multiply': (f'Calculate {a} * {b}. Respond with the integer only.', a*b),
            'subtract_oracle_product': (f'Calculate {a*b} - {c}. Respond with the integer only.', a*b-c),
            'compound_expression': (f'Calculate ({a} * {b}) - {c}. Respond with the integer only.', a*b-c),
            'prose_oracle_product': (f'There are {a*b} cups in total. Take away {c} cups. How many cups remain? Reply only with the integer.', a*b-c),
            'explicit_intermediate': (f'{a} trays hold {b} cups each, making {a*b} cups in total. Then {c} cups are removed. Reply only with the number remaining.', a*b-c),
        }
        for mode,(prompt,expected) in modes.items():
            rows.append({'id':f'arithmetic-{i:02d}-{mode}','family':'arithmetic','case':i,
                         'mode':mode,'prompt':prompt,'expected':str(expected)})
        a,b,c=f'R{i+201}',f'S{i+301}',f'T{i+401}'
        for earliest in (True,False):
            requested='earliest' if earliest else 'latest'
            # Compare wording while keeping event roles fixed; permutation changes
            # names across roles, not which temporal rule is correct.
            modes={
                'chronology': (f'Chronology: {c} follows {b}, which follows {a}. Select the {requested} of the three. Output only its identifier.',a if earliest else c),
                'training_template': (f'{a} occurs before {b}; {c} occurs after {b}. Give only the identifier of the {"first" if earliest else "last"} event.',a if earliest else c),
                'explicit_chain': (f'Temporal order from first to last: {a}, {b}, {c}. Which event is {requested}? Reply with its identifier only.',a if earliest else c),
                'permuted_roles': (f'Chronology: {a} follows {c}, which follows {b}. Select the {requested} of the three. Output only its identifier.',b if earliest else a),
            }
            for mode,(prompt,expected) in modes.items():
                rows.append({'id':f'order-{i:02d}-{mode}-{requested}','family':'order','case':i,
                    'mode':mode,'requested':requested,'prompt':prompt,'expected':expected})
    return rows


def extract_answer(row,response):
    text=response.strip()
    if row['family']=='arithmetic':
        if re.fullmatch(r'-?\d+',text):return text
        boxes=re.findall(r'\\boxed\{(-?\d+)\}',text)
        if boxes and len(set(boxes))==1:return boxes[0]
        return None
    # Narrow whole-response extraction; do not search for a favorable event ID.
    match=re.fullmatch(r'(?:Event\s+)?([RST]\d+)\.?',text)
    return match.group(1) if match else None


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cases=probes()
    (output/'probes.json').write_text(json.dumps(cases,indent=2))
    device=choose_device(args.device)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16).to(device).eval()
    result={'scope':'diagnostic paired interventions, not held-out benchmark','arms':{}}
    config=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
    for arm in ('baseline','remediation'):
        if arm=='remediation':
            checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            install_lora(model,checkpoint['config']['rank'],checkpoint['config']['alpha'])
            load_adapter(model,checkpoint['adapter'])
        rows=[]
        for case in cases:
            inputs=chat_inputs(tokenizer,case['prompt']).to(device)
            with torch.inference_mode():
                ids=model.generate(**inputs,generation_config=config)[0,inputs['input_ids'].shape[1]:].tolist()
            response=tokenizer.decode(ids,skip_special_tokens=True)
            parsed=extract_answer(case,response)
            rows.append({**case,'response':response,'completion_ids':ids,'parsed_answer':parsed,
                'strict':response.strip()==case['expected'],'content':parsed==case['expected'] if parsed is not None else None})
        result['arms'][arm]=rows
        (output/'rules.json').write_text(json.dumps(result,indent=2))
        for family in ('arithmetic','order'):
            for mode in sorted({r['mode'] for r in rows if r['family']==family}):
                selected=[r for r in rows if r['family']==family and r['mode']==mode]
                print(arm,family,mode,'strict',sum(r['strict'] for r in selected),'content',sum(r['content'] is True for r in selected),'n',len(selected),flush=True)
    result['complete']=True
    (output/'rules.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
