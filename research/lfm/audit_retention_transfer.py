#!/usr/bin/env python3
"""Independent artifact checks for the fixed retention-transfer experiment."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from research.lfm.native_lora import supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.native_retention_transfer import ARMS, SEEDS, digest, normalize, verify


def audit_data(root, source):
    freeze = verify(root)
    groups = json.loads((root/'target.json').read_text())
    original = json.loads((source/'contrast.json').read_text())
    if digest(source/'contrast.json') != freeze['source_sha256'] or groups != original[:128]:
        raise ValueError('Target training not identical to registered source pass')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    facts = set()
    for group in groups:
        if len(group['rows']) != 4 or sum(r['supervised_tokens'] for r in group['rows']) != 8:
            raise ValueError('Target accumulation mismatch')
        for row in group['rows']:
            a, b, c = row['operands']
            answer = str(a*b + (-c if row['operation'] == 'subtract' else c))
            if row['response'] != answer:
                raise ValueError('Wrong training arithmetic')
            encoded = supervised_chat(tokenizer, row['prompt'], answer, 512)
            if any(row[k] != v for k, v in encoded.items()):
                raise ValueError('Training tokens/masks mismatch')
            facts.add(tuple(row['operands']))
    protected = json.loads((root/'protected.json').read_text())
    if len(protected) != 4 or len({r['id'] for r in protected}) != 4:
        raise ValueError('Protection example IDs mismatch')
    for row in protected:
        encoded = supervised_chat(tokenizer, row['prompt'], row['response'], 512)
        if any(row[k] != v for k, v in encoded.items()):
            raise ValueError('Protection tokens/masks mismatch')
    tasks = json.loads((root/'tasks.json').read_text())['tasks']
    historical = set()
    for path, expected in freeze['historical_suites'].items():
        if digest(path) != expected:
            raise ValueError('Historical task source changed')
        artifact = json.loads(Path(path).read_text())
        historical.update(normalize(t['prompt']) for t in artifact['tasks']
                          if isinstance(t, dict) and 'prompt' in t)
    train_prompts = {normalize(r['prompt']) for g in groups for r in g['rows']}
    protected_prompts = {normalize(r['prompt']) for r in protected}
    for task in tasks:
        if normalize(task['prompt']) in historical | train_prompts | protected_prompts:
            raise ValueError('Evaluation leakage by exact normalized prompt')
        if task['category'] == 'arithmetic':
            if tuple(task['operands']) in facts:
                raise ValueError('Heldout arithmetic fact appears in training')
            a, b, c = task['operands']
            if task['expected'] != str(a*b+(-c if task['operation'] == 'subtract' else c)):
                raise ValueError('Wrong heldout arithmetic label')
    return {'target_groups': len(groups), 'unique_training_facts': len(facts),
            'protected_examples': len(protected), 'tasks': len(tasks),
            'historical_suites_checked': len(freeze['historical_suites']),
            'normalized_prompt_overlap': 0, 'arithmetic_train_fact_overlap': 0,
            'tokenization_and_masks_verified': True,
            'limit': 'Exact overlap audit, not semantic near-duplicate or independent-blinding guarantee'}


def audit_training(root):
    reports = {}
    frozen_hashes, seed_initial_hashes = set(), []
    for seed in SEEDS:
        initial_hashes, budgets = set(), set()
        for arm in ARMS:
            name = f'{arm}-seed-{seed}'
            path = root/name/'training.json'
            artifact = json.loads(path.read_text())
            config = artifact['config']
            if not artifact['complete'] or config['seed'] != seed or config['arm'] != arm:
                raise ValueError('Training arm/seed mismatch')
            if config['freeze_sha256'] != digest(root/'freeze.json'):
                raise ValueError('Training freeze mismatch')
            history = artifact['history']
            if [h['step'] for h in history] != list(range(1, 129)):
                raise ValueError('Training endpoint mismatch')
            if artifact['frozen_before_sha256'] != artifact['frozen_after_sha256']:
                raise ValueError('Frozen base mutation')
            frozen_hashes.add(artifact['frozen_before_sha256'])
            checkpoint = torch.load(root/name/'checkpoint.pt', map_location='cpu', weights_only=True)
            if checkpoint['config'] != config:
                raise ValueError('Checkpoint/training config mismatch')
            if sum(t.numel() for t in checkpoint['adapter'].values()) != artifact['trainable_parameters']:
                raise ValueError('Checkpoint parameter count mismatch')
            if any(not torch.isfinite(t).all() for t in checkpoint['adapter'].values()):
                raise ValueError('Nonfinite checkpoint')
            initial_hashes.add(artifact['initial_adapter_sha256'])
            budgets.add(tuple(artifact[k] for k in ('target_sequences', 'target_supervised_tokens',
                'target_input_tokens', 'retention_gradient_sequences', 'retention_gradient_input_tokens',
                'retention_gradient_supervised_tokens', 'retention_measurement_forward_sequences')))
            for h in history:
                geometry = h['geometry']
                if geometry['raw_norm'] and not geometry['zero_projection']:
                    if abs(geometry['raw_norm'] - geometry['projected_norm']) > 1e-10:
                        raise ValueError('Projection norm mismatch')
                    if abs(geometry['raw_norm'] - geometry['random_norm']) > 1e-10:
                        raise ValueError('Random norm mismatch')
                if max(map(abs, geometry['gradient_dots_projected']), default=0) > 1e-8:
                    raise ValueError('Projection did not remove protected gradient span')
            reports[name] = {'training_sha256': digest(path),
                'checkpoint_sha256': digest(root/name/'checkpoint.pt'),
                'zero_projection_steps': sum(h['geometry']['zero_projection'] for h in history),
                'max_absolute_ideal_projected_dot': max(abs(d) for h in history
                    for d in h['geometry']['gradient_dots_projected']),
                'max_absolute_realized_dot': max(abs(d) for h in history for d in h['realized_gradient_dots']),
                'retention_ce_increase_steps_per_example': [sum(h['retention_ce_after'][i] >
                    h['retention_ce_before'][i] for h in history) for i in range(4)]}
        if len(initial_hashes) != 1 or len(budgets) != 1:
            raise ValueError('Same-seed initial weights/processing budgets differ')
        seed_initial_hashes.extend(initial_hashes)
    if len(frozen_hashes) != 1 or len(set(seed_initial_hashes)) != len(SEEDS):
        raise ValueError('Different seeds not distinct or base weights differ across arms')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'data': audit_data(args.root, args.source)}
    if args.training:
        result['training'] = audit_training(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
