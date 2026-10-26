#!/usr/bin/env python3
"""Same-token-history adapter on/off causal forks with nonnumeric donor prefixes."""

import argparse
from contextlib import contextmanager
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.audit_native_arithmetic_access import (
    REASONING_REQUEST, numeric_support, score_response,
)
from research.lfm.evaluate_pretrained_development import chat_inputs, synchronize
from research.lfm.native_lora import LoRALinear, install_lora, load_adapter
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.native_retention_transfer import digest, save
from research.lfm.probe_lfm import choose_device


PROTOCOL = Path('research/reports/native_same_history_protocol.md')
CARDINALS = frozenset(('zero one two three four five six seven eight nine ten eleven twelve '
                      'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty '
                      'thirty forty fifty sixty seventy eighty ninety hundred thousand million '
                      'billion trillion dozen').split())


def unsafe_text(text):
    if any(c.isnumeric() for c in text):
        return 'numeric_character'
    if CARDINALS.intersection(re.findall(r'[a-z]+', text.casefold())):
        return 'cardinal_number_word'
    return None


def safe_prefix(tokenizer, donor_ids, cap=16):
    prefix = []
    specials = set(tokenizer.all_special_ids)
    full = tokenizer.decode(donor_ids, skip_special_tokens=False)
    boundaries = [(i, 'numeric_character') for i, c in enumerate(full) if c.isnumeric()]
    boundaries.extend((m.start(), 'cardinal_number_word') for m in
                      re.finditer(r'[a-z]+', full.casefold()) if m.group() in CARDINALS)
    boundary = min(boundaries, default=None)
    stop, rejected = 'donor_end', None
    for token in donor_ids:
        if len(prefix) >= cap:
            stop = 'token_cap'
            break
        if token in specials:
            stop, rejected = 'special_token', token
            break
        candidate = tokenizer.decode(prefix+[token], skip_special_tokens=False)
        # Full-donor lexical boundary prevents leaking a cardinal split over
        # BPE tokens, including a partial word at the sixteen-token cap.
        reason = boundary[1] if boundary is not None and len(candidate) > boundary[0] else unsafe_text(candidate)
        if reason:
            stop, rejected = reason, token
            break
        prefix.append(token)
    return {'ids': prefix, 'text': tokenizer.decode(prefix, skip_special_tokens=False),
            'stop_reason': stop, 'rejected_token_id': rejected,
            'rejected_token_text': tokenizer.decode([rejected]) if rejected is not None else None}


@contextmanager
def adapter_mode(model, active):
    modules = {m: m.scale for m in model.modules() if isinstance(m, LoRALinear)}
    if not modules:
        raise ValueError('No adapters installed')
    try:
        if not active:
            for module in modules:
                module.scale = 0.
        yield
    finally:
        for module, scale in modules.items():
            module.scale = scale


def distribution_pair(off, on):
    """CPU full next-token distributions at precisely the same input history."""
    a, b = off.double().log_softmax(-1), on.double().log_softmax(-1)
    pa, pb = a.exp(), b.exp()
    middle = torch.logaddexp(a, b)-torch.log(torch.tensor(2., dtype=torch.float64))
    oi, ni = int(a.argmax()), int(b.argmax())
    return {'off_top1_id': oi, 'on_top1_id': ni, 'top1_equal': oi == ni,
            'off_top1_probability_under_off': float(pa[oi]),
            'off_top1_probability_under_on': float(pb[oi]),
            'on_top1_probability_under_off': float(pa[ni]),
            'on_top1_probability_under_on': float(pb[ni]),
            'js_nats': float(((pa*(a-middle)).sum()+(pb*(b-middle)).sum())/2)}


def first_divergence(a, b):
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def pair_changes(rows, reference):
    old = {r['id']: r for r in reference}
    if len(old) != len(reference) or len(rows) != len(reference) or {r['id'] for r in rows} != set(old):
        raise ValueError('Mismatched/duplicate task IDs')
    return {'gains': [r['id'] for r in rows if r['scores']['confirmed_content'] and
                     not old[r['id']]['scores']['confirmed_content']],
            'losses': [r['id'] for r in rows if not r['scores']['confirmed_content'] and
                      old[r['id']]['scores']['confirmed_content']]}


def summarize(records, pairs, executions):
    summaries = {}
    for seed in (0, 1, 2):
        for mode in ('original', 'reasoning_request'):
            group = [r for r in records if r['seed'] == seed and r['prompt_mode'] == mode]
            baseline = {state: [r for r in group if r['donor'] == 'empty' and r['recipient'] == state]
                        for state in ('off', 'on')}
            for donor in ('empty', 'native', 'adapted'):
                for state in ('off', 'on'):
                    rows = [r for r in group if r['donor'] == donor and r['recipient'] == state]
                    key = f'seed-{seed}/{mode}/{donor}/{state}'
                    summaries[key] = {'n': len(rows),
                        'confirmed_content': sum(r['scores']['confirmed_content'] for r in rows),
                        'native_exact': sum(r['scores']['native_exact'] for r in rows),
                        'bare_integer_format': sum(r['scores']['bare_integer_format'] for r in rows),
                        'truncated': sum(r['hit_token_limit_without_eos'] for r in rows),
                        'nonempty_prefixes': sum(bool(r['prefix']['ids']) for r in rows),
                        'self_replay_checked': sum(r['self_replay_parity'] is not None for r in rows),
                        'self_replay_mismatches': [r['id'] for r in rows if r['self_replay_parity'] is False],
                        'vs_own_empty': pair_changes(rows, baseline[state]),
                        'vs_off_same_history': pair_changes(rows, [r for r in group if r['donor'] == donor and r['recipient'] == 'off'])}
                selected = [p for p in pairs if p['seed'] == seed and p['prompt_mode'] == mode and p['donor'] == donor]
                summaries[f'seed-{seed}/{mode}/{donor}/pair'] = {
                    'n': len(selected), 'same_next_token': sum(p['distribution']['top1_equal'] for p in selected),
                    'mean_js_nats': sum(p['distribution']['js_nats'] for p in selected)/len(selected),
                    'same_continuation': sum(p['first_continuation_divergence'] is None for p in selected)}
    return {'conditions': summaries, 'case_conditions': len(records), 'paired_histories': len(pairs),
            'unique_executions': len(executions), 'aliased_case_conditions': len(records)-len(executions),
            'actual_unique_execution_seconds': sum(e['runtime_seconds'] for e in executions.values()),
            'actual_unique_generated_tokens': sum(len(e['generated_token_ids']) for e in executions.values())}


def run(args):
    root, source, output = args.checkpoints, args.access, args.output
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((source/'access.json').read_text())
    if not previous['complete']:
        raise ValueError('Incomplete donor experiment')
    tasks_path = root/'tasks.json'
    if digest(tasks_path) != previous['manifest']['tasks_sha256']:
        raise ValueError('Task source changed')
    tasks = [t for t in json.loads(tasks_path.read_text())['tasks'] if t['category'] == 'arithmetic']
    if len(tasks) != 40 or len({t['id'] for t in tasks}) != 40:
        raise ValueError('Expected all forty cases')
    if previous['manifest']['reasoning_request'] != REASONING_REQUEST:
        raise ValueError('Reasoning request changed')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    support = numeric_support(tokenizer)
    device = choose_device(args.device)
    dtype = torch.float16 if device != 'cpu' else torch.float32
    manifest = {'protocol_sha256': digest(PROTOCOL), 'implementation_sha256': digest(Path(__file__)),
        'source_access_sha256': digest(source/'access.json'), 'tasks_sha256': digest(tasks_path),
        'checkpoint_sha256': {str(seed): digest(root/f'ordinary-seed-{seed}'/'checkpoint.pt') for seed in (0, 1, 2)},
        'scope': 'exposed diagnostic, no training; identical tokens not identical hidden states',
        'prefix_cap': 16, 'total_assistant_budget': 192,
        'prefix_rule_cardinals': sorted(CARDINALS)}
    for seed in (0, 1, 2):
        if manifest['checkpoint_sha256'][str(seed)] != previous['manifest']['checkpoints_sha256'][f'ordinary-seed-{seed}']:
            raise ValueError('Checkpoint differs from recorded donor')
    save(output/'freeze.json', manifest)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype).to(device).eval()
    targets = None
    records, pairs, executions = [], [], {}
    for seed in (0, 1, 2):
        name = f'ordinary-seed-{seed}'
        checkpoint = torch.load(root/name/'checkpoint.pt', map_location='cpu', weights_only=True)
        config = checkpoint['config']
        if config['model'] != MODEL or config['revision'] != REVISION or config['seed'] != seed or config['arm'] != 'ordinary':
            raise ValueError('Checkpoint config mismatch')
        if targets is None:
            targets = install_lora(model, config['rank'], config['alpha'])
        if targets != config['targets']:
            raise ValueError('Adapter targets differ')
        load_adapter(model, checkpoint['adapter'])
        cache, logits_cache = {}, {}
        for mode in ('original', 'reasoning_request'):
            donors = {label: {r['id']: r for r in previous['arms'][arm][mode]}
                      for label, arm in (('native', 'baseline'), ('adapted', name))}
            for task in tasks:
                prompt = task['prompt']+('\n\n'+REASONING_REQUEST if mode == 'reasoning_request' else '')
                prompt_ids = chat_inputs(tokenizer, prompt)['input_ids'][0].tolist()
                for donor in ('empty', 'native', 'adapted'):
                    prefix = ({'ids': [], 'text': '', 'stop_reason': 'empty_control',
                               'rejected_token_id': None, 'rejected_token_text': None} if donor == 'empty'
                              else safe_prefix(tokenizer, donors[donor][task['id']]['generated_token_ids']))
                    if unsafe_text(prefix['text']):
                        raise ValueError('Unsafe supplied prefix')
                    history_ids = prompt_ids+prefix['ids']
                    rows, logits_pair = {}, {}
                    for state in ('off', 'on'):
                        key = (state, tuple(history_ids))
                        reused = key in cache
                        if not reused:
                            inputs = torch.tensor([history_ids], device=device)
                            generation = GenerationConfig(do_sample=False, num_beams=1,
                                repetition_penalty=1., max_new_tokens=192-len(prefix['ids']),
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id, use_cache=True)
                            with adapter_mode(model, state == 'on'):
                                synchronize(device)
                                start = time.perf_counter()
                                with torch.inference_mode():
                                    logits = model(inputs, use_cache=False).logits[0, -1].float().cpu()
                                    ids = model.generate(inputs, generation_config=generation)[0, len(history_ids):].tolist()
                                synchronize(device)
                                elapsed = time.perf_counter()-start
                            execution_id = f'e{len(executions):04d}'
                            execution = {'execution_id': execution_id, 'seed': seed, 'recipient': state,
                                'input_token_ids': history_ids, 'generated_token_ids': ids,
                                'runtime_seconds': elapsed, 'next_top1_id': int(logits.argmax()),
                                'eos_probability': float(logits.softmax(-1)[tokenizer.eos_token_id]),
                                'numeric_support_probability_mass': float(logits.softmax(-1)[support].sum())}
                            executions[execution_id] = execution
                            cache[key], logits_cache[key] = execution_id, logits
                        execution_id = cache[key]
                        execution, logits = executions[execution_id], logits_cache[key]
                        generated = execution['generated_token_ids']
                        response = tokenizer.decode(prefix['ids']+generated, skip_special_tokens=True)
                        self_source = ('native' if state == 'off' else 'adapted')
                        stored = donors[self_source][task['id']]['generated_token_ids']
                        self_check = generated == stored[len(prefix['ids']):] if donor in ('empty', self_source) else None
                        row = {'id': task['id'], 'seed': seed, 'prompt_mode': mode, 'prompt': prompt,
                            'donor': donor, 'prefix': prefix, 'recipient': state,
                            'execution_id': execution_id, 'execution_alias': reused,
                            'response': response, 'scores': score_response(task, response),
                            'continuation_scores': score_response(task, tokenizer.decode(generated, skip_special_tokens=True)),
                            'ended_with_eos': bool(generated and generated[-1] == tokenizer.eos_token_id),
                            'hit_token_limit_without_eos': len(prefix['ids'])+len(generated) >= 192 and
                                                         generated[-1] != tokenizer.eos_token_id,
                            'self_replay_parity': self_check}
                        if donor == 'empty' and not self_check:
                            save(output/'control_failure.json', {'case': row, 'expected_ids': stored, 'actual_ids': generated})
                            raise RuntimeError('Empty-prefix engineering control failed')
                        records.append(row)
                        rows[state], logits_pair[state] = row, logits
                    off, on = [executions[rows[s]['execution_id']] for s in ('off', 'on')]
                    if off['input_token_ids'] != on['input_token_ids']:
                        raise ValueError('On/off input histories differ')
                    pairs.append({'id': task['id'], 'seed': seed, 'prompt_mode': mode, 'donor': donor,
                        'off_execution_id': off['execution_id'], 'on_execution_id': on['execution_id'],
                        'distribution': distribution_pair(logits_pair['off'], logits_pair['on']),
                        'first_continuation_divergence': first_divergence(off['generated_token_ids'], on['generated_token_ids'])})
            print('seed', seed, mode, 'complete', flush=True)
        save(output/f'seed-{seed}.json', {'records': [r for r in records if r['seed'] == seed],
                                        'pairs': [p for p in pairs if p['seed'] == seed]})
        del logits_cache, cache
    if digest(PROTOCOL) != manifest['protocol_sha256'] or digest(Path(__file__)) != manifest['implementation_sha256']:
        raise ValueError('Protocol or implementation changed during run')
    if digest(source/'access.json') != manifest['source_access_sha256']:
        raise ValueError('Donor data changed during run')
    for seed in (0, 1, 2):
        if digest(root/f'ordinary-seed-{seed}'/'checkpoint.pt') != manifest['checkpoint_sha256'][str(seed)]:
            raise ValueError('Checkpoint changed during run')
    summary = summarize(records, pairs, executions)
    save(output/'forks.json', {'complete': True, 'manifest': manifest, 'model': MODEL,
        'revision': REVISION, 'dtype': str(dtype), 'device': device,
        'records': records, 'pairs': pairs, 'executions': executions, 'summary': summary})
    save(output/'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--access', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto', choices=('auto', 'mps', 'cuda', 'cpu'))
    run(parser.parse_args())
