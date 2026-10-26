#!/usr/bin/env python3
"""Replay real AdamW updates and fork matched norm-preserving retention controls."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

from research.lfm.evaluate_pretrained_development import chat_inputs,content_validation
from research.lfm.native_lora import adapter_state,install_lora,load_adapter
from research.lfm.native_lora_smoke import MODEL,REVISION
from research.lfm.probe_lfm import choose_device


def remove_harmful_component(delta,retention_gradient):
    """Euclidean half-space projection; same-norm variant isolates direction."""
    if delta.ndim!=1 or delta.shape!=retention_gradient.shape:
        raise ValueError('Update and retention gradient must be matching flat vectors')
    if not torch.isfinite(delta).all() or not torch.isfinite(retention_gradient).all():
        raise ValueError('Nonfinite displacement or retention gradient')
    delta=delta.double();gradient=retention_gradient.double()
    dot=float(delta@gradient);norm2=float(gradient@gradient)
    projected=delta-gradient*(max(0.,dot)/norm2) if norm2>0 else delta.clone()
    projected_norm=float(projected.norm());original_norm=float(delta.norm())
    matched=(projected*(original_norm/projected_norm)
             if projected_norm>0 else torch.zeros_like(delta))
    return projected,matched,{'retention_gradient_dot_raw_update':dot,
        'raw_update_norm':original_norm,'retention_gradient_norm':norm2**.5,
        'removed_component_norm':float((delta-projected).norm()),
        'projection_active':dot>0 and norm2>0,
        'retention_gradient_dot_projected':float(gradient@projected),
        'norm_matched_update_norm':float(matched.norm())}


def flatten(parameters):
    return torch.cat([p.detach().float().cpu().flatten() for p in parameters])


def assign(parameters,flat):
    if flat.ndim!=1 or len(flat)!=sum(p.numel() for p in parameters) or not torch.isfinite(flat).all():
        raise ValueError('Invalid parameter vector; no parameters assigned')
    start=0
    with torch.no_grad():
        for parameter in parameters:
            size=parameter.numel()
            value=flat[start:start+size].view_as(parameter).to(device=parameter.device,dtype=parameter.dtype)
            parameter.copy_(value)
            start+=size
    if start!=len(flat):raise ValueError('Displacement shape mismatch')


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    data=Path(args.data);raw=(data/'contrast.json').read_bytes();groups=json.loads(raw)
    manifest=json.loads((data/'manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=manifest['arms']['contrast']['sha256']:
        raise ValueError('Source data changed')
    expected_checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    tasks=json.loads((data/'tasks.json').read_text())['tasks'];by_id={t['id']:t for t in tasks}
    native=json.loads((Path(args.evaluations)/'native_semantic_transfer_v1_baseline/baseline.json').read_text())
    native_rows={r['id']:r for r in native['results']}
    device=choose_device(args.device)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    torch.manual_seed(0)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32).to(device).eval()
    targets=install_lora(model,8,16)
    if expected_checkpoint['config']['targets']!=targets:raise ValueError('Source targets differ')
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    protection_id='record-0';key_cases=('record-0','record-1','record-4')
    fixed_prefixes={};target=tokenizer.encode('label',add_special_tokens=False)
    if len(target)!=1:raise ValueError('Expected one key token')
    target=target[0]
    for name in key_cases:
        prompt=chat_inputs(tokenizer,by_id[name]['prompt']).to(device)
        native_ids=native_rows[name]['generated_token_ids']
        if native_ids[4]!=target:raise ValueError('Native reference prefix differs')
        fixed_prefixes[name]=torch.cat((prompt['input_ids'],torch.tensor([native_ids[:4]],device=device)),dim=1)
    def key_loss(name):
        logits=model(fixed_prefixes[name],use_cache=False).logits[0,-1].float()
        return F.cross_entropy(logits[None],torch.tensor([target],device=device))
    def group_loss(group):
        total=0.
        for row in group['rows']:
            ids=torch.tensor([row['input_ids']],device=device)
            labels=torch.tensor([row['labels']],device=device)
            total=total+model(ids,labels=labels,use_cache=False).loss*(row['supervised_tokens']/8)
        return total
    generation=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.,max_new_tokens=192,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
    probe_ids=key_cases+('new_facts_new_template-00','counterfactual_add-00')
    result={'scope':'same-state one-update causal forks on raw contrast training replay; no candidate model approval',
        'protected_case':protection_id,'unprotected_key_cases':['record-1','record-4'],
        'data_sha256':hashlib.sha256(raw).hexdigest(),
        'source_checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'checkpoints_measured':[1,16,64,128,256,512],
        'projection':'Euclidean LoRA-factor half-space; one schema-key CE gradient, not full retention objective',
        'scale16_caveat':'extrapolated displacement, not an actual optimizer step or local guarantee',
        'boundaries':[]}
    for step,group in enumerate(groups,1):
        measured=step in result['checkpoints_measured']
        if measured:
            before=flatten(parameters)
            protected=key_loss(protection_id)
            gradients=torch.autograd.grad(protected,parameters)
            ret_gradient=torch.cat([g.detach().float().cpu().flatten() for g in gradients])
            del protected,gradients
        optimizer.zero_grad(set_to_none=True)
        loss_value=0.
        for row in group['rows']:
            ids=torch.tensor([row['input_ids']],device=device)
            labels=torch.tensor([row['labels']],device=device)
            loss=model(ids,labels=labels,use_cache=False).loss*(row['supervised_tokens']/8)
            loss.backward();loss_value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        if not measured:continue
        raw_after=flatten(parameters);delta=raw_after-before
        projected,matched,stats=remove_harmful_component(delta,ret_gradient)
        variants={'zero':torch.zeros_like(delta),'raw':delta,'projected':projected,
                  'projected_norm_matched':matched,'raw_x16':delta*16,
                  'projected_norm_matched_x16':matched*16}
        boundary={'step':step,'preupdate_target_loss':loss_value,
            'unclipped_gradient_norm':float(norm),'direction_measurements':stats,'forks':[]}
        try:
            for name,displacement in variants.items():
                assign(parameters,before.double()+displacement.double())
                realized=flatten(parameters)-before
                with torch.inference_mode():
                    actual_target=float(group_loss(group))
                    actual_keys={case:float(key_loss(case)) for case in key_cases}
                responses=[]
                for case in probe_ids:
                    task=by_id[case];inputs=chat_inputs(tokenizer,task['prompt']).to(device)
                    with torch.inference_mode():
                        ids=model.generate(**inputs,generation_config=generation)[0,inputs['input_ids'].shape[1]:].tolist()
                    response=tokenizer.decode(ids,skip_special_tokens=True)
                    responses.append({'id':case,'response':response,'completion_ids':ids,
                        'content':content_validation(task,response)['pass'],
                        'native_exact':response.strip()==task['expected'] if task['validator']=='text_exact' else None})
                boundary['forks'].append({'variant':name,'actual_target_ce':actual_target,
                    'actual_schema_key_ce':actual_keys,
                    'realized_retention_gradient_dot_displacement':float(ret_gradient.double()@realized.double()),
                    'realized_displacement_norm':float(realized.double().norm()),'responses':responses})
        finally:
            assign(parameters,raw_after)
        boundary['raw_parameters_restored_exactly']=torch.equal(flatten(parameters),raw_after)
        result['boundaries'].append(boundary)
        (output/'updates.json').write_text(json.dumps(result,indent=2))
        print(step,json.dumps(stats),flush=True)
    actual=adapter_state(model)
    error=max(float((actual[name]-expected_checkpoint['adapter'][name]).abs().max()) for name in actual)
    result['raw_training_replication_max_adapter_error']=error
    result['raw_replay_replication_passed']=error==0.
    result['complete']=True
    (output/'updates.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint','evaluations','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
