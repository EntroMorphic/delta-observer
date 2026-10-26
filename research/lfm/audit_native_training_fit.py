#!/usr/bin/env python3
"""Measure exact in-sample generation fit of native rule-remediation checkpoints."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.evaluate_pretrained_development import chat_inputs
from research.lfm.native_lora import install_lora, load_adapter
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.probe_lfm import choose_device


def run(args):
    rows=json.loads(Path(args.data).read_text())
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    device=choose_device(args.device)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16).to(device).eval()
    install_lora(model,checkpoint['config']['rank'],checkpoint['config']['alpha'])
    load_adapter(model,checkpoint['adapter'])
    config=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=64,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
    out={'scope':'in-sample generation diagnostic, not generalization','modes':{},'rows':[]}
    modes = getattr(args, 'modes', None) or (
        'multiply','subtract','compound_final','chronology','relations','chain','inverted')
    for mode in modes:
        selected=[row for row in rows if row.get('mode')==mode]
        for row in selected:
            inputs=chat_inputs(tokenizer,row['prompt']).to(device)
            with torch.inference_mode():
                ids=model.generate(**inputs,generation_config=config)[0,inputs['input_ids'].shape[1]:].tolist()
            response=tokenizer.decode(ids,skip_special_tokens=True)
            out['rows'].append({'mode':mode,'prompt':row['prompt'],'expected':row['response'],
                               'response':response,'strict':response.strip()==row['response']})
        out['modes'][mode]={'examples':len(selected),
            'strict_passes':sum(row['strict'] for row in out['rows'] if row['mode']==mode)}
        print(mode,out['modes'][mode],flush=True)
    with Path(args.output).open('x') as destination:
        json.dump(out,destination,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    parser.add_argument('--modes',nargs='+',help='Explicit preregistered training modes to audit')
    run(parser.parse_args())
