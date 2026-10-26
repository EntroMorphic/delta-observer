#!/usr/bin/env python3
"""Diagnostic full-trajectory norm-matched projection against one native key loss."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

from research.lfm.audit_native_update_conflict import assign,flatten,remove_harmful_component
from research.lfm.evaluate_pretrained_development import chat_inputs,synchronize
from research.lfm.native_lora import adapter_state,install_lora
from research.lfm.native_lora_smoke import MODEL,REVISION,frozen_digest
from research.lfm.probe_lfm import choose_device


def run(args):
    data=Path(args.data);raw_manifest=(data/'manifest.json').read_bytes();manifest=json.loads(raw_manifest)
    raw=(data/'contrast.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['arms']['contrast']['sha256']:raise ValueError('Source changed')
    groups=json.loads(raw);tasks=json.loads((data/'tasks.json').read_text())['tasks']
    protected=next(t for t in tasks if t['id']=='record-0')
    native_path=Path(args.native_evaluation)
    native=json.loads(native_path.read_text())
    source=next(r for r in native['results'] if r['id']=='record-0')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    key=tokenizer.encode('label',add_special_tokens=False)
    if len(key)!=1 or source['generated_token_ids'][4]!=key[0]:raise ValueError('Reference key differs')
    device=choose_device(args.device);torch.manual_seed(0)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32).to(device).eval()
    targets=install_lora(model,8,16);parameters=[p for p in model.parameters() if p.requires_grad]
    prefix=chat_inputs(tokenizer,protected['prompt']).to(device)
    fixed=torch.cat((prefix['input_ids'],torch.tensor([source['generated_token_ids'][:4]],device=device)),dim=1)
    before_hash=frozen_digest(model)
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    def retention_loss():
        return F.cross_entropy(model(fixed,use_cache=False).logits[0,-1].float()[None],
                               torch.tensor(key,device=device))
    config={'model':MODEL,'revision':REVISION,'rank':8,'alpha':16,'targets':targets,
        'arm':'contrast_projected_diagnostic','seed':0,'dtype':'float32',
        'manifest_sha256':hashlib.sha256(raw_manifest).hexdigest(),'data_sha256':hashlib.sha256(raw).hexdigest(),
        'optimizer':'AdamW 1e-4 decay .01 clip 1; actual post-optimizer displacement projected at unchanged norm',
        'constraint':'one exposed native schema-key CE, record-0; not a blind retention claim',
        'protected_native_evaluation_sha256':hashlib.sha256(native_path.read_bytes()).hexdigest(),
        'optimizer_state_policy':'keep Adam moments from actual target gradients; only realized parameter delta is projected'}
    history=[];synchronize(device);start=time.perf_counter()
    for step,group in enumerate(groups,1):
        old=flatten(parameters);ret_loss=retention_loss()
        gradients=torch.autograd.grad(ret_loss,parameters)
        g=torch.cat([t.detach().float().cpu().flatten() for t in gradients])
        key_before=float(ret_loss.detach());del ret_loss,gradients
        optimizer.zero_grad(set_to_none=True);loss_value=0.
        for row in group['rows']:
            ids=torch.tensor([row['input_ids']],device=device)
            labels=torch.tensor([row['labels']],device=device)
            loss=model(ids,labels=labels,use_cache=False).loss*(row['supervised_tokens']/8)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite target loss')
            loss.backward();loss_value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step();raw_new=flatten(parameters)
        projected,matched,stats=remove_harmful_component(raw_new-old,g)
        assign(parameters,old.double()+matched)
        with torch.no_grad():key_after=float(retention_loss())
        if not torch.isfinite(torch.tensor(key_after)):raise RuntimeError('Nonfinite retention loss')
        history.append({'step':step,'target_ce_before':loss_value,'key_ce_before':key_before,
            'key_ce_after':key_after,'raw_gradient_norm':float(norm),
            'projection':stats,'supervised_tokens':sum(r['supervised_tokens'] for r in group['rows']),
            'input_tokens':sum(len(r['input_ids']) for r in group['rows'])})
        if step%16==0 or step==len(groups):
            temporary=output/'checkpoint.pending.pt'
            torch.save({'config':config,'adapter':adapter_state(model),
                'optimizer':optimizer.state_dict(),'history':history},temporary)
            temporary.replace(output/'checkpoint.pt')
            print(json.dumps(history[-1]),flush=True)
    synchronize(device);elapsed=time.perf_counter()-start;after_hash=frozen_digest(model)
    report={'config':config,'history':history,'steps':len(history),
        'new_update_seconds':elapsed,'frozen_before_sha256':before_hash,'frozen_after_sha256':after_hash,
        'trainable_parameters':sum(p.numel() for p in parameters),
        'supervised_tokens':sum(h['supervised_tokens'] for h in history),
        'input_tokens':sum(h['input_tokens'] for h in history),
        'projection_steps':sum(h['projection']['projection_active'] for h in history),
        'scope':'mechanism test with one exposed protection case, not general-purpose model training'}
    (output/'training.json').write_text(json.dumps(report,indent=2))
    if before_hash!=after_hash:raise RuntimeError('Frozen base changed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','native-evaluation','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
