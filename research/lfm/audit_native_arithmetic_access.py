#!/usr/bin/env python3
"""Inference-only arithmetic access controls; no target-dependent generation."""

import argparse
import gc
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.evaluate_pretrained_development import (
    chat_inputs, content_validation, synchronize, validate,
)
from research.lfm.native_lora import LoRALinear, install_lora, load_adapter
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.native_retention_transfer import digest, save
from research.lfm.probe_lfm import choose_device


PROTOCOL = Path('research/reports/native_arithmetic_access_protocol.md')
ANSWER_PREFIX = 'Final answer: '
REASONING_REQUEST = (r'For this diagnostic, replace the answer-format instruction above: '
                     r'show the arithmetic steps and finish with a single \boxed{integer}.')


def numeric_support(tokenizer):
    special = set(tokenizer.all_special_ids)
    support = []
    for token in range(len(tokenizer)):
        if token in special:
            continue
        piece = tokenizer.decode([token], skip_special_tokens=False)
        if (re.fullmatch(r'-?[0-9]+', piece) and str(int(piece)) == piece
                and tokenizer.encode(piece, add_special_tokens=False) == [token]):
            support.append(token)
    if not support:
        raise ValueError('Empty numeric support')
    return support


def numeric_choice(logits, support):
    """No task/answer parameter: support is whole tokenizer-vocabulary support."""
    index = torch.tensor(support, device=logits.device)
    return int(index[logits[index].argmax()])


def readout(logits, target):
    logits = logits.float()
    return {'target_token_id': target, 'target_probability': float(logits.softmax(-1)[target]),
            'target_rank': int(1+(logits > logits[target]).sum()),
            'top1_id': int(logits.argmax()), 'target_vs_top1_logit': float(logits[target]-logits.max())}


def score_response(task, response):
    exact = validate(task, response)['pass']
    content = content_validation(task, response)
    return {'native_exact': exact, 'confirmed_content': exact or content['pass'] is True,
            'content_validation': content, 'bare_integer_format': bool(re.fullmatch(r'-?[0-9]+', response.strip()))}


def paired(rows, reference):
    expected = {r['id']: r for r in reference}
    if len(expected) != len(reference) or len(rows) != len(reference) or {r['id'] for r in rows} != set(expected):
        raise ValueError('Mismatched or duplicate IDs')
    return {'content_gains': [r['id'] for r in rows if r['scores']['confirmed_content'] and
                             not expected[r['id']]['scores']['confirmed_content']],
            'content_losses': [r['id'] for r in rows if not r['scores']['confirmed_content'] and
                              expected[r['id']]['scores']['confirmed_content']]}


def summarize(arms):
    summaries = {}
    for arm, conditions in arms.items():
        summaries[arm] = {}
        for condition, rows in conditions.items():
            summary = {'n': len(rows),
                'native_exact': sum(r['scores']['native_exact'] for r in rows),
                'confirmed_content': sum(r['scores']['confirmed_content'] for r in rows),
                'bare_integer_format': sum(r['scores']['bare_integer_format'] for r in rows),
                'truncated': sum(r['hit_token_limit_without_eos'] for r in rows),
                'generated_tokens': sum(len(r['generated_token_ids']) for r in rows),
                'runtime_seconds': sum(r['runtime_seconds'] for r in rows),
                'vs_own_original': paired(rows, conditions['original']),
                'vs_native_original': paired(rows, arms['baseline']['original'])}
            if condition == 'answer_prefix':
                summary['continuation_only_exact_diagnostic'] = sum(
                    r['continuation_scores']['native_exact'] for r in rows)
                summary['continuation_only_content_diagnostic'] = sum(
                    r['continuation_scores']['confirmed_content'] for r in rows)
            summaries[arm][condition] = summary
    return summaries


def run(args):
    root, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((root/'freeze.json').read_text())
    if digest(root/'tasks.json') != freeze['files']['tasks.json']:
        raise ValueError('Source test changed')
    tasks = [t for t in json.loads((root/'tasks.json').read_text())['tasks'] if t['category'] == 'arithmetic']
    if len(tasks) != 40 or len({t['id'] for t in tasks}) != 40:
        raise ValueError('Expected all forty exposed cases')
    names = ['baseline']+[f'ordinary-seed-{seed}' for seed in (0, 1, 2)]
    sources = {name: root/'eval'/name/'baseline.json' for name in names}
    originals = {name: json.loads(path.read_text()) for name, path in sources.items()}
    for artifact in originals.values():
        if not artifact['complete'] or artifact['tasks_sha256'] != digest(root/'tasks.json'):
            raise ValueError('Incomplete or mismatched source evaluation')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    support = numeric_support(tokenizer)
    target_ids = {}
    # This only checks registered control support. Neither mask nor decoder gets answers.
    for task in tasks:
        ids = tokenizer.encode(task['expected'], add_special_tokens=False)
        if len(ids) != 1 or ids[0] not in support:
            raise ValueError('Registered numeric control unsupported: abort without filtering')
        target_ids[task['id']] = ids[0]
    manifest = {'protocol_sha256': digest(PROTOCOL), 'implementation_sha256': digest(Path(__file__)),
        'scope': 'post-opening diagnostic, no training or new confirmatory test',
        'tasks_sha256': digest(root/'tasks.json'),
        'source_evaluations_sha256': {n: digest(p) for n, p in sources.items()},
        'checkpoints_sha256': {n: digest(root/n/'checkpoint.pt') for n in names if n != 'baseline'},
        'numeric_support_ids': support, 'numeric_support_size': len(support),
        'numeric_support_sha256': hashlib.sha256(json.dumps(support).encode()).hexdigest(),
        'answer_prefix': ANSWER_PREFIX, 'reasoning_request': REASONING_REQUEST,
        'names': names, 'conditions': ['original', 'numeric_decoder', 'answer_prefix', 'reasoning_request'],
        'adapter_off_control': 'all three ordinary seeds, original prompts'}
    save(output/'freeze.json', manifest)
    device = choose_device(args.device)
    dtype = torch.float16 if device != 'cpu' else torch.float32
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype).to(device).eval()
    generation = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.,
        max_new_tokens=192, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, use_cache=True)
    for name, artifact in originals.items():
        if (artifact['model'] != MODEL or artifact['revision'] != REVISION
                or artifact['dtype'] != str(dtype) or artifact['generation_config'] != generation.to_dict()):
            raise ValueError('Source generation settings differ')
        if name != 'baseline' and artifact['adapter']['sha256'] != manifest['checkpoints_sha256'][name]:
            raise ValueError('Source adapter provenance differs')
    payload = {'complete': False, 'manifest': manifest, 'model': MODEL, 'revision': REVISION,
               'device': device, 'dtype': str(dtype), 'generation_config': generation.to_dict(), 'arms': {}}
    for name in names:
        if name != 'baseline':
            checkpoint_path = root/name/'checkpoint.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            config = checkpoint['config']
            if config['model'] != MODEL or config['revision'] != REVISION or config['arm'] != 'ordinary':
                raise ValueError('Checkpoint mismatch')
            if name == names[1]:
                targets = install_lora(model, config['rank'], config['alpha'])
            if config['targets'] != targets:
                raise ValueError('Adapter target mismatch')
            load_adapter(model, checkpoint['adapter'])
            if digest(checkpoint_path) != manifest['checkpoints_sha256'][name]:
                raise ValueError('Checkpoint changed during inference')
        stored = {r['id']: r for r in originals[name]['results']}
        native = {r['id']: r for r in originals['baseline']['results']}
        conditions = {c: [] for c in manifest['conditions']}
        if name != 'baseline':
            conditions['all_adapters_off'] = []
        for condition, rows in conditions.items():
            for task in tasks:
                prompt = task['prompt']
                if condition == 'reasoning_request':
                    prompt += '\n\n'+REASONING_REQUEST
                inputs = chat_inputs(tokenizer, prompt).to(device)
                prefill = ANSWER_PREFIX if condition == 'answer_prefix' else ''
                if prefill:
                    extra = tokenizer.encode(prefill, add_special_tokens=False)
                    inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor([extra], device=device)), dim=1)
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                scales = {}
                if condition == 'all_adapters_off':
                    for module in model.modules():
                        if isinstance(module, LoRALinear):
                            scales[module] = module.scale
                            module.scale = 0.
                synchronize(device)
                start = time.perf_counter()
                try:
                    with torch.inference_mode():
                        logits = model(**inputs, use_cache=False).logits[0, -1].float()
                        if condition == 'numeric_decoder':
                            ids = [numeric_choice(logits, support), tokenizer.eos_token_id]
                        else:
                            ids = model.generate(**inputs, generation_config=generation)[0, inputs['input_ids'].shape[1]:].tolist()
                    synchronize(device)
                    seconds = time.perf_counter()-start
                finally:
                    for module, scale in scales.items():
                        module.scale = scale
                continuation = tokenizer.decode(ids, skip_special_tokens=True)
                full_response = prefill+continuation
                row = {'id': task['id'], 'condition': condition, 'prompt': prompt,
                       'prefill': prefill, 'input_token_ids': inputs['input_ids'][0].tolist(),
                       'generated_token_ids': ids, 'continuation': continuation, 'response': full_response,
                       'scores': score_response(task, full_response),
                       'continuation_scores': score_response(task, continuation),
                       'first_token': readout(logits, target_ids[task['id']]),
                       'numeric_support_probability_mass': float(logits.softmax(-1)[support].sum()),
                       'runtime_seconds': seconds, 'ended_with_eos': ids[-1] == tokenizer.eos_token_id,
                       'hit_token_limit_without_eos': len(ids) >= 192 and ids[-1] != tokenizer.eos_token_id}
                if condition in ('original', 'all_adapters_off'):
                    expected = native[task['id']] if condition == 'all_adapters_off' else stored[task['id']]
                    row['stored_generation_token_parity'] = ids == expected['generated_token_ids']
                    if not row['stored_generation_token_parity']:
                        save(output/'parity_failure.json', {'arm': name, 'case': row,
                                                           'expected_ids': expected['generated_token_ids']})
                        raise RuntimeError('Original/control token parity failed; diagnostic aborted')
                rows.append(row)
            print(name, condition, 'content', sum(r['scores']['confirmed_content'] for r in rows), flush=True)
        payload['arms'][name] = conditions
        save(output/f'{name}.json', {'complete': True, 'conditions': conditions})
    if digest(PROTOCOL) != manifest['protocol_sha256'] or digest(Path(__file__)) != manifest['implementation_sha256']:
        raise RuntimeError('Protocol/implementation changed during inference')
    payload['complete'] = True
    payload['summary'] = summarize(payload['arms'])
    save(output/'access.json', payload)
    save(output/'summary.json', payload['summary'])
    del model
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto', choices=('auto', 'mps', 'cuda', 'cpu'))
    run(parser.parse_args())
