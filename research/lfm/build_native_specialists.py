#!/usr/bin/env python3
"""Remove other-task updates while preserving each task's v2 exposure sequence."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def subset(rows, history, category):
    indices = [i for i, row in enumerate(rows) if row['category'] == category]
    remap = {old: new for new, old in enumerate(indices)}
    seen = [record['row_index'] for record in history]
    if any(type(i) is not int for i in seen) or sorted(seen) != list(range(len(rows))):
        raise ValueError('Source history must expose each source row exactly once')
    order = [remap[i] for i in seen if i in remap]
    if not order:
        raise ValueError('Requested task has no examples')
    return [rows[i] for i in indices], order


def build(source, training, output):
    manifest_bytes = (source/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    spec = manifest['arms']['remediation']
    raw = (source/spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('Source data hash mismatch')
    trained = json.loads(training.read_text())
    if (trained['config']['data_sha256'] != spec['sha256'] or
            trained['config']['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError('Source history/data provenance mismatch')
    if trained['frozen_before_sha256'] != trained['frozen_after_sha256']:
        raise ValueError('Source base changed')
    rows = json.loads(raw)
    output.mkdir(parents=True, exist_ok=False)
    for label, category in [('arithmetic', 'arithmetic'), ('ordering', 'transitive_ordering')]:
        selected, order = subset(rows, trained['history'], category)
        directory = output/label
        directory.mkdir()
        raw = (json.dumps(selected, indent=2, ensure_ascii=False)+'\n').encode()
        (directory/'remediation.json').write_bytes(raw)
        new_spec = {'path':'remediation.json', 'sha256':hashlib.sha256(raw).hexdigest(),
            'examples':len(selected), 'training_order':order,
            'input_tokens':sum(len(r['input_ids']) for r in selected),
            'supervised_tokens':sum(r['supervised_tokens'] for r in selected),
            'mode_counts':dict(Counter(r['mode'] for r in selected))}
        new_manifest = {'model':manifest['model'], 'revision':manifest['revision'],
            'arms':{'remediation':new_spec}, 'source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
            'source_training_sha256':hashlib.sha256(training.read_bytes()).hexdigest(),
            'scope':'task-only exposure-order-matched ablation; not equal total update/compute budget'}
        (directory/'manifest.json').write_text(json.dumps(new_manifest, indent=2))
        print(label, {k:v for k,v in new_spec.items() if k != 'training_order'}, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','training','output'):parser.add_argument(f'--{key}',required=True,type=Path)
    args=parser.parse_args()
    build(args.source,args.training,args.output)
