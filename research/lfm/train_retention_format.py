#!/usr/bin/env python3
"""Token-weighted microbatch accumulation for equal-token native LoRA comparisons."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.evaluate_pretrained_development import synchronize
from research.lfm.native_lora import adapter_state, install_lora, supervised_chat
from research.lfm.native_lora_smoke import MODEL,REVISION,frozen_digest
from research.lfm.probe_lfm import choose_device


def verify_groups(groups,tokenizer,budget):
    for group in groups:
        if (group.get('supervised_tokens')!=budget
                or sum(r['supervised_tokens'] for r in group['rows'])!=budget):
            raise ValueError('Unequal supervised-token update budget')
        for row in group['rows']:
            verified=supervised_chat(tokenizer,row['prompt'],row['response'],128)
            if any(row[k]!=v for k,v in verified.items()):raise ValueError('Tokenization/mask mismatch')


def run(args):
    directory=Path(args.data);raw_manifest=(directory/'manifest.json').read_bytes()
    manifest=json.loads(raw_manifest);spec=manifest['arms'][args.arm]
    raw=(directory/spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Data hash mismatch')
    if manifest['model']!=MODEL or manifest['revision']!=REVISION:raise ValueError('Wrong base')
    groups=json.loads(raw)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    budget=manifest['per_update_supervised_tokens'];verify_groups(groups,tokenizer,budget)
    if len(groups)!=spec['updates']:raise ValueError('Update count mismatch')
    for key,actual in [('supervised_tokens',sum(r['supervised_tokens'] for g in groups for r in g['rows'])),
                       ('input_tokens',sum(len(r['input_ids']) for g in groups for r in g['rows'])),
                       ('sequence_exposures',sum(len(g['rows']) for g in groups))]:
        if actual!=spec[key]:raise ValueError(f'Manifest budget mismatch: {key}')
    device=choose_device(args.device);torch.manual_seed(0)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32).to(device).eval()
    targets=install_lora(model,8,16);before=frozen_digest(model)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    config={'model':MODEL,'revision':REVISION,'rank':8,'alpha':16,'targets':targets,'arm':args.arm,
        'seed':0,'dtype':'float32','lr':1e-4,'weight_decay':.01,'gradient_clip':1.,
        'manifest_sha256':hashlib.sha256(raw_manifest).hexdigest(),'data_sha256':spec['sha256'],
        'loss':'supervised-token-weighted CE across sequential microbatches','supervised_tokens_per_update':budget}
    history=[];synchronize(device);started=time.perf_counter()
    for index,group in enumerate(groups):
        optimizer.zero_grad(set_to_none=True);loss_value=0.
        for row in group['rows']:
            ids=torch.tensor([row['input_ids']],device=device)
            labels=torch.tensor([row['labels']],device=device)
            loss=model(ids,labels=labels,use_cache=False).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            weighted=loss*(row['supervised_tokens']/budget)
            weighted.backward();loss_value+=float(weighted.detach())
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        history.append({'step':index+1,'loss':loss_value,'gradient_norm':float(norm),
            'input_tokens':sum(len(r['input_ids']) for r in group['rows']),
            'supervised_tokens':budget,'microbatches':len(group['rows'])})
        if (index+1)%16==0 or index+1==len(groups):
            temporary=output/'checkpoint.pending.pt'
            torch.save({'config':config,'adapter':adapter_state(model),'optimizer':optimizer.state_dict(),
                        'history':history},temporary)
            temporary.replace(output/'checkpoint.pt')
            print(json.dumps(history[-1]),flush=True)
    synchronize(device);elapsed=time.perf_counter()-started;after=frozen_digest(model)
    report={'config':config,'steps':len(history),'history':history,
        'trainable_parameters':sum(p.numel() for p in parameters),
        'new_update_seconds':elapsed,'input_tokens':sum(r['input_tokens'] for r in history),
        'supervised_tokens':sum(r['supervised_tokens'] for r in history),
        'frozen_before_sha256':before,'frozen_after_sha256':after,
        'scope':manifest.get('scope','one-seed exploratory comparison; input lengths and compute differ')}
    (output/'training.json').write_text(json.dumps(report,indent=2))
    if before!=after:raise RuntimeError('Frozen base changed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--arm',choices=('final_only','worked_replay','extra_final',
                                       'repeat','paraphrase','contrast'),required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
