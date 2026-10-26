#!/usr/bin/env python3
"""Exact selected-direction Hessian products; no quantization or stochastic diagonal."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.native_lora import install_lora, load_adapter, supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.probe_lfm import choose_device


def directional_derivatives(loss, parameters, direction):
    gradients = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True)
    first = sum((g * d).sum() for g, d in zip(gradients, direction))
    products = torch.autograd.grad(first, parameters, retain_graph=True, allow_unused=True)
    second = sum((h * d).sum() for h, d in zip(products, direction) if h is not None)
    return float(first.detach()), float(second.detach())


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['config']['model'] != MODEL or checkpoint['config']['revision'] != REVISION:
        raise ValueError('Checkpoint base revision mismatch')
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    # Eager attention is intentional: fused attention may lack double backward.
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        dtype=torch.float32, attn_implementation='eager').to(device).eval()
    targets = install_lora(model, checkpoint['config']['rank'], checkpoint['config']['alpha'])
    if targets != checkpoint['config']['targets']:
        raise ValueError('Checkpoint target modules differ')
    load_adapter(model, checkpoint['adapter'])
    named = [(name,p) for name,p in model.named_parameters() if name.endswith('lora_b')]
    parameters = [p for _,p in named]
    norms = {}
    directions = {}
    for group in ('all', 'attention', 'conv'):
        raw = [p.detach().clone() if (group=='all' or
            (group=='attention' and '.self_attn.' in name) or
            (group=='conv' and '.conv.' in name)) else torch.zeros_like(p)
            for name,p in named]
        norm = sum(float(v.square().sum()) for v in raw)**.5
        if norm == 0:
            raise ValueError('Cannot normalize a zero update direction')
        directions[group] = [v/norm for v in raw]
        norms[group] = norm
    source = json.loads(Path(args.data).read_text())
    # Fixed first six compound rows: neither chosen by loss nor by test success.
    selected = [row for row in source if row.get('mode')=='compound_final'][:6]
    result = {'scope':'exact FP32 Hessian directional derivatives on six in-sample CE losses',
        'directions':'unit-norm current LoRA B tensors, A held fixed; groups all/attention/conv',
        'not_a_full_hessian_or_quantization_map':True, 'no_quantization_performed':True,
        'inference_speed_or_capability_gain_claimed':False, 'attention_implementation':'eager',
        'parameterization_caveat':'LoRA-factor curvature is not native effective-weight curvature and depends on factor scaling',
        'checkpoint':args.checkpoint, 'direction_norms_before_normalization':norms,
        'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'finite_difference_epsilon':.01,'rows':[]}
    originals = [p.detach().clone() for p in parameters]
    for row in selected:
        encoded=supervised_chat(tokenizer,row['prompt'],row['response'],128)
        inputs=torch.tensor([encoded['input_ids']],device=device)
        labels=torch.tensor([encoded['labels']],device=device)
        loss=model(inputs,labels=labels,use_cache=False).loss
        measurements={group:dict(zip(('gradient_dot_direction','directional_curvature'),
            directional_derivatives(loss,parameters,direction)))
            for group,direction in directions.items()}
        baseline=float(loss.detach())
        del loss
        # Validate directional derivative with actual symmetric perturbations.
        values=[]
        try:
            for sign in (-1,1):
                with torch.no_grad():
                    for p,original,d in zip(parameters,originals,directions['all']):
                        p.copy_(original+sign*.01*d)
                    values.append(float(model(inputs,labels=labels,use_cache=False).loss))
        finally:
            with torch.no_grad():
                for p,original in zip(parameters,originals):p.copy_(original)
        record={'prompt':row['prompt'],'response':row['response'],'ce':baseline,
            'directions':measurements,'actual_ce_minus_epsilon':values[0],
            'actual_ce_plus_epsilon':values[1],
            'symmetric_fd_first':(values[1]-values[0])/.02,
            'symmetric_fd_second':(values[1]+values[0]-2*baseline)/.0001,
            'finite_difference_note':'finite-step validation, not claimed exact Hessian entries'}
        result['rows'].append(record)
        (output/'curvature.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(record),flush=True)
    result['perturbations_restored_exactly']=all(torch.equal(p,original) for p,original in zip(parameters,originals))
    result['complete']=True
    (output/'curvature.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','data','output'):parser.add_argument(f'--{key}',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
