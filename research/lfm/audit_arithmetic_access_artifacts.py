#!/usr/bin/env python3
"""Reproduce access-diagnostic scoring, input construction, parity and paired losses."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from research.lfm.audit_native_arithmetic_access import (
    ANSWER_PREFIX, REASONING_REQUEST, numeric_support, paired, score_response, summarize,
)
from research.lfm.evaluate_pretrained_development import chat_inputs
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.native_retention_transfer import digest, save


def audit(root, source):
    payload = json.loads((root/'access.json').read_text())
    manifest = json.loads((root/'freeze.json').read_text())
    if not payload['complete'] or manifest != payload['manifest']:
        raise ValueError('Incomplete/mismatched diagnostic')
    if digest(source/'tasks.json') != manifest['tasks_sha256']:
        raise ValueError('Source tasks changed')
    if digest(Path('research/reports/native_arithmetic_access_protocol.md')) != manifest['protocol_sha256']:
        raise ValueError('Protocol changed')
    if digest(Path('research/lfm/audit_native_arithmetic_access.py')) != manifest['implementation_sha256']:
        raise ValueError('Inference implementation changed')
    tasks = {t['id']: t for t in json.loads((source/'tasks.json').read_text())['tasks']
             if t['category'] == 'arithmetic'}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    support = numeric_support(tokenizer)
    if support != manifest['numeric_support_ids']:
        raise ValueError('Numeric support does not reproduce')
    originals = {}
    for name, expected in manifest['source_evaluations_sha256'].items():
        path = source/'eval'/name/'baseline.json'
        if digest(path) != expected:
            raise ValueError('Source evaluation changed')
        originals[name] = {r['id']: r for r in json.loads(path.read_text())['results']}
    for name, expected in manifest['checkpoints_sha256'].items():
        if digest(source/name/'checkpoint.pt') != expected:
            raise ValueError('Checkpoint changed')
    checked, parity = 0, 0
    conditional = {}
    for name, conditions in payload['arms'].items():
        individual = json.loads((root/f'{name}.json').read_text())
        if not individual['complete'] or individual['conditions'] != conditions:
            raise ValueError('Individual result differs from consolidated artifact')
        for condition, rows in conditions.items():
            if len(rows) != 40 or {r['id'] for r in rows} != set(tasks):
                raise ValueError('Missing/duplicate diagnostic case')
            for row in rows:
                task = tasks[row['id']]
                prompt = task['prompt'] + ('\n\n'+REASONING_REQUEST if condition == 'reasoning_request' else '')
                prefix = ANSWER_PREFIX if condition == 'answer_prefix' else ''
                ids = chat_inputs(tokenizer, prompt)['input_ids'][0].tolist()
                ids += tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
                if prompt != row['prompt'] or ids != row['input_token_ids'] or prefix != row['prefill']:
                    raise ValueError('Unexpected generation input; no calculated hints permitted')
                generated = row['generated_token_ids']
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if text != row['continuation'] or prefix+text != row['response']:
                    raise ValueError('Generation/response mismatch')
                if row['scores'] != score_response(task, prefix+text):
                    raise ValueError('Full-response score mismatch')
                if row['continuation_scores'] != score_response(task, text):
                    raise ValueError('Conditional diagnostic score mismatch')
                if condition == 'numeric_decoder' and (len(generated) != 2 or
                        generated[0] not in support or generated[1] != tokenizer.eos_token_id):
                    raise ValueError('Unexpected numeric-control support/length')
                if condition in ('original', 'all_adapters_off'):
                    reference = originals['baseline' if condition == 'all_adapters_off' else name][row['id']]
                    if generated != reference['generated_token_ids'] or not row['stored_generation_token_parity']:
                        raise ValueError('Stored token parity does not reproduce')
                    parity += 1
                checked += 1
        prefix_rows = [{**r, 'scores': r['continuation_scores']} for r in conditions['answer_prefix']]
        conditional[name] = {'scope': 'Continuation-only prefix diagnostic, NOT native exact compliance',
            'continuation_exact': sum(r['scores']['native_exact'] for r in prefix_rows),
            'vs_own_original': paired(prefix_rows, conditions['original']),
            'vs_native_original': paired(prefix_rows, payload['arms']['baseline']['original'])}
    if summarize(payload['arms']) != payload['summary']:
        raise ValueError('Consolidated summary does not reproduce')
    return {'complete': True, 'checked_case_conditions': checked, 'verified_token_parity_cases': parity,
            'numeric_support_size': len(support), 'inputs_tokens_and_frozen_scores_reproduce': True,
            'source_checkpoint_hashes_unchanged': True, 'conditional_prefix_pairs': conditional,
            'artifact_sha256': digest(root/'access.json'),
            'scope': 'Independent artifact reconstruction, not an independent model or blinded evaluation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.source)
    save(args.output, result)
    print(json.dumps(result, indent=2))
