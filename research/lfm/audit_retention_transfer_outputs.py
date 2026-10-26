#!/usr/bin/env python3
"""Post-unsealing diagnostics; never changes the frozen transfer scores."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from research.lfm.native_retention_transfer import ARMS, SEEDS, digest
from research.lfm.score_native_holdout100 import score


def classify_integer(task, response):
    """Whole-response integer taxonomy, not a replacement content-success parser."""
    if not re.fullmatch(r'-?\d+', response.strip()):
        return 'not_bare_integer'
    value = int(response.strip())
    a, b, c = task['operands']
    expected = a*b + (-c if task['operation'] == 'subtract' else c)
    if value == expected:
        return 'correct_bare_integer'
    if value == a*b:
        return 'unchanged_product'
    if value == a*b + (c if task['operation'] == 'subtract' else -c):
        return 'opposite_operation'
    return 'other_wrong_integer'


def audit(root):
    freeze = json.loads((root/'freeze.json').read_text())
    if digest(root/'tasks.json') != freeze['files']['tasks.json']:
        raise ValueError('Frozen test changed')
    tasks = json.loads((root/'tasks.json').read_text())['tasks']
    names = ['baseline']+[f'{arm}-seed-{seed}' for seed in SEEDS for arm in ARMS]
    summary = json.loads((root/'comparison.json').read_text())
    result = {'scope': 'Post-unsealing diagnostic classification, no scoring changes or new confirmatory claims',
              'tasks_sha256': digest(root/'tasks.json'), 'arms': {}}
    for name in names:
        path = root/'eval'/name/'baseline.json'
        artifact = json.loads(path.read_text())
        if not artifact['complete'] or artifact['tasks_sha256'] != freeze['files']['tasks.json']:
            raise ValueError('Incomplete or unmatched evaluation')
        if digest(path) != summary['evaluation_sources'][name]:
            raise ValueError('Evaluation changed after scoring')
        rescored = score(tasks, artifact['results'])
        if rescored['tasks'] != summary['arms'][name]['tasks']:
            raise ValueError('Stored scores do not reproduce')
        rows = {r['id']: r for r in artifact['results']}
        classes = Counter()
        examples = []
        for task in tasks:
            if task['category'] != 'arithmetic':
                continue
            row = rows[task['id']]
            kind = classify_integer(task, row['response'])
            classes[kind] += 1
            examples.append({'id': task['id'], 'operands': task['operands'],
                'operation': task['operation'], 'expected': task['expected'],
                'response': row['response'], 'class': kind,
                'frozen_validation': row['validation'], 'frozen_content_validation': row['content_validation']})
        same_answer = sum(rows[f'fresh-subtract-{i:02d}']['response'].strip() ==
                          rows[f'fresh-add-{i:02d}']['response'].strip() for i in range(20))
        raw = artifact['results']
        result['arms'][name] = {'arithmetic_response_classes': dict(classes),
            'same_full_response_operation_pairs': same_answer,
            'truncated_cases': [r['id'] for r in raw if r['hit_token_limit_without_eos']],
            'generated_tokens': sum(r['generated_tokens'] for r in raw),
            'evaluation_seconds': sum(r['elapsed_seconds'] for r in raw),
            'arithmetic_examples': examples}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open('x') as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps({name: {k: v for k, v in arm.items() if k != 'arithmetic_examples'}
                      for name, arm in result['arms'].items()}, indent=2))
