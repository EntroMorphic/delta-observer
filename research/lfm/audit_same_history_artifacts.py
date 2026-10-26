#!/usr/bin/env python3
"""Independently reconstruct same-history inputs, scores, aliases and replay controls."""

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from research.lfm.audit_native_arithmetic_access import REASONING_REQUEST, score_response
from research.lfm.audit_native_same_history import (
    PROTOCOL, first_divergence, safe_prefix, summarize, unsafe_text,
)
from research.lfm.evaluate_pretrained_development import chat_inputs
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.native_retention_transfer import digest, save


def audit(root, access, checkpoints):
    payload = json.loads((root/'forks.json').read_text())
    manifest = json.loads((root/'freeze.json').read_text())
    if not payload['complete'] or payload['manifest'] != manifest:
        raise ValueError('Incomplete/mismatched experiment')
    if digest(PROTOCOL) != manifest['protocol_sha256']:
        raise ValueError('Protocol changed')
    if digest(Path('research/lfm/audit_native_same_history.py')) != manifest['implementation_sha256']:
        raise ValueError('Inference implementation changed')
    if digest(access/'access.json') != manifest['source_access_sha256']:
        raise ValueError('Donor data changed')
    if digest(checkpoints/'tasks.json') != manifest['tasks_sha256']:
        raise ValueError('Tasks changed')
    for seed, expected in manifest['checkpoint_sha256'].items():
        if digest(checkpoints/f'ordinary-seed-{seed}'/'checkpoint.pt') != expected:
            raise ValueError('Checkpoint changed')
    donor_source = json.loads((access/'access.json').read_text())
    tasks = {t['id']: t for t in json.loads((checkpoints/'tasks.json').read_text())['tasks']
             if t['category'] == 'arithmetic'}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    records, pairs, executions = payload['records'], payload['pairs'], payload['executions']
    index, unique_keys = {}, set()
    self_checks, self_failures, empty_checks = 0, [], 0
    prefix_counts, stops = {}, {}
    for execution_id, execution in executions.items():
        if execution['execution_id'] != execution_id:
            raise ValueError('Execution key mismatch')
        key = (execution['seed'], execution['recipient'], tuple(execution['input_token_ids']))
        if key in unique_keys:
            raise ValueError('Duplicate execution not recorded as alias')
        unique_keys.add(key)
        if not math.isfinite(execution['runtime_seconds']) or execution['runtime_seconds'] < 0:
            raise ValueError('Invalid execution timing')
    seen_executions = set()
    for row in records:
        seed, mode, donor, state = row['seed'], row['prompt_mode'], row['donor'], row['recipient']
        key = (seed, mode, donor, state, row['id'])
        if key in index:
            raise ValueError('Duplicate case-condition')
        index[key] = row
        task = tasks[row['id']]
        prompt = task['prompt']+('\n\n'+REASONING_REQUEST if mode == 'reasoning_request' else '')
        source_arm = 'baseline' if donor == 'native' else f'ordinary-seed-{seed}'
        donor_rows = {r['id']: r for r in donor_source['arms'][source_arm][mode]}
        if donor == 'empty':
            if row['prefix']['ids'] or row['prefix']['text']:
                raise ValueError('Nonempty empty-prefix control')
        else:
            expected = safe_prefix(tokenizer, donor_rows[row['id']]['generated_token_ids'])
            if row['prefix'] != expected:
                raise ValueError('Prefix extraction/source provenance mismatch')
        if unsafe_text(row['prefix']['text']) or len(row['prefix']['ids']) > 16:
            raise ValueError('Number/control-length failure in supplied prefix')
        if set(row['prefix']['ids']) & set(tokenizer.all_special_ids):
            raise ValueError('Prefix includes control token')
        history = chat_inputs(tokenizer, prompt)['input_ids'][0].tolist()+row['prefix']['ids']
        execution = executions[row['execution_id']]
        if history != execution['input_token_ids'] or prompt != row['prompt']:
            raise ValueError('Input histories do not reconstruct')
        if execution['seed'] != seed or execution['recipient'] != state:
            raise ValueError('Aliased across seed/adapter-state')
        if row['execution_alias'] != (row['execution_id'] in seen_executions):
            raise ValueError('Alias flag does not reconstruct')
        seen_executions.add(row['execution_id'])
        generated = execution['generated_token_ids']
        response = tokenizer.decode(row['prefix']['ids']+generated, skip_special_tokens=True)
        if response != row['response'] or row['scores'] != score_response(task, response):
            raise ValueError('Full response/score mismatch')
        if row['continuation_scores'] != score_response(task, tokenizer.decode(generated, skip_special_tokens=True)):
            raise ValueError('Continuation diagnostic mismatch')
        if len(row['prefix']['ids'])+len(generated) > 192:
            raise ValueError('Output exceeds registered total budget')
        own_source = 'baseline' if state == 'off' else f'ordinary-seed-{seed}'
        stored = next(r['generated_token_ids'] for r in donor_source['arms'][own_source][mode]
                      if r['id'] == row['id'])
        expected_self = generated == stored[len(row['prefix']['ids']):] if donor in (
            'empty', 'native' if state == 'off' else 'adapted') else None
        if row['self_replay_parity'] != expected_self:
            raise ValueError('Self-replay flag does not reproduce')
        if expected_self is not None:
            self_checks += 1
            if not expected_self:
                self_failures.append(list(key))
        if donor == 'empty':
            empty_checks += 1
            if not expected_self:
                raise ValueError('Original engineering parity failed')
        if state == 'on':
            group = f'seed-{seed}/{mode}/{donor}'
            prefix_counts.setdefault(group, {})
            prefix_counts[group][row['prefix']['text']] = prefix_counts[group].get(row['prefix']['text'], 0)+1
            stops[row['prefix']['stop_reason']] = stops.get(row['prefix']['stop_reason'], 0)+1
    if len(records) != 1440 or len(pairs) != 720 or len(seen_executions) != len(executions):
        raise ValueError('Registered coverage mismatch')
    for seed in (0, 1, 2):
        individual = json.loads((root/f'seed-{seed}.json').read_text())
        if individual != {'records': [r for r in records if r['seed'] == seed],
                          'pairs': [p for p in pairs if p['seed'] == seed]}:
            raise ValueError('Per-seed artifact mismatch')
        for mode in ('original', 'reasoning_request'):
            for donor in ('empty', 'native', 'adapted'):
                for state in ('off', 'on'):
                    if {key[-1] for key in index if key[:4] == (seed, mode, donor, state)} != set(tasks):
                        raise ValueError('Missing/extra case in registered condition')
    paired_keys = set()
    for pair in pairs:
        seed, mode, donor, identifier = pair['seed'], pair['prompt_mode'], pair['donor'], pair['id']
        key = (seed, mode, donor, identifier)
        if key in paired_keys:
            raise ValueError('Duplicate paired history')
        paired_keys.add(key)
        left = index[(seed, mode, donor, 'off', identifier)]
        right = index[(seed, mode, donor, 'on', identifier)]
        off, on = [executions[r['execution_id']] for r in (left, right)]
        if off['input_token_ids'] != on['input_token_ids']:
            raise ValueError('On/off history mismatch')
        if pair['off_execution_id'] != left['execution_id'] or pair['on_execution_id'] != right['execution_id']:
            raise ValueError('Pair provenance mismatch')
        if pair['first_continuation_divergence'] != first_divergence(off['generated_token_ids'], on['generated_token_ids']):
            raise ValueError('Continuation divergence mismatch')
        probability = pair['distribution']
        if probability['off_top1_id'] != off['next_top1_id'] or probability['on_top1_id'] != on['next_top1_id']:
            raise ValueError('Next-token top-1 mismatch')
        if not 0 <= probability['js_nats'] <= math.log(2)+1e-12:
            raise ValueError('Invalid recorded JS')
    expected_summary = summarize(records, pairs, executions)
    if expected_summary != payload['summary'] or expected_summary != json.loads((root/'summary.json').read_text()):
        raise ValueError('Summary does not reproduce')
    return {'complete': True, 'case_conditions_reconstructed': len(records), 'on_off_histories_identical': len(pairs),
            'unique_executions': len(executions), 'alias_records_verified': len(records)-len(executions),
            'self_replays_verified': self_checks, 'self_replay_failures': self_failures,
            'empty_prefix_parity_cases': empty_checks, 'prefix_text_counts': prefix_counts,
            'prefix_stop_reasons': stops, 'inputs_and_scores_reproduce': True,
            'forks_sha256': digest(root/'forks.json'),
            'limit': 'Artifact reconstruction; full next-token distributions were not persisted, so JS is range/provenance-checked, not independently recomputed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'access', 'checkpoints', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.access, args.checkpoints)
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'prefix_text_counts'}, indent=2))
