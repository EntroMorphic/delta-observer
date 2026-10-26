#!/usr/bin/env python3
"""Locked split/pair and retention scoring for the semantic-transfer comparison."""

import argparse
import hashlib
import json
from pathlib import Path

from research.lfm.score_native_holdout100 import counts, score


def run(args):
    data=Path(args.data);raw=(data/'tasks.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    frozen=json.loads((data/'freeze.json').read_text())
    manifest_bytes=(data/'manifest.json').read_bytes()
    if (digest!=frozen['tasks_sha256'] or
        hashlib.sha256(manifest_bytes).hexdigest()!=frozen['training_manifest_sha256']):
        raise ValueError('Frozen artifacts changed')
    tasks=json.loads(raw)['tasks'];splits={t['id']:t['split'] for t in tasks}
    arms={};sources={};reference=None
    for name in ('baseline','repeat','paraphrase','contrast'):
        path=Path(args.evaluations)/f'native_semantic_transfer_v1_{name}/baseline.json'
        raw_eval=path.read_bytes();artifact=json.loads(raw_eval)
        if not artifact['complete'] or artifact['tasks_sha256']!=digest:
            raise ValueError('Incomplete/unmatched evaluation')
        settings={k:artifact[k] for k in ('model','revision','dtype','generation_config')}
        if reference is None:reference=settings
        elif reference!=settings:raise ValueError('Generation settings differ')
        if name!='baseline':
            config=artifact['adapter']['config']
            if config['arm']!=name or config['manifest_sha256']!=frozen['training_manifest_sha256']:
                raise ValueError('Adapter arm/provenance mismatch')
        arm=score(tasks,artifact['results'])
        arm['splits']={split:counts([r for r in arm['tasks'] if splits[r['id']]==split])
                       for split in sorted(set(splits.values()))}
        by_id={r['id']:r for r in arm['tasks']}
        arm['both_wordings_correct']=sum(by_id[f'new_facts_seen_template-{i:02d}']['strict']
            and by_id[f'new_facts_new_template-{i:02d}']['strict'] for i in range(20))
        arm['both_operations_correct']=sum(by_id[f'new_facts_new_template-{i:02d}']['strict']
            and by_id[f'counterfactual_add-{i:02d}']['strict'] for i in range(20))
        arms[name]=arm;sources[name]={'path':str(path),'sha256':hashlib.sha256(raw_eval).hexdigest()}
    base={r['id']:r for r in arms['baseline']['tasks']}
    repeat={r['id']:r for r in arms['repeat']['tasks']}
    for name in ('repeat','paraphrase','contrast'):
        arm=arms[name]
        arm['baseline_exact_losses']=[r['id'] for r in arm['tasks'] if base[r['id']]['strict'] and not r['strict']]
        arm['baseline_content_losses']=[r['id'] for r in arm['tasks'] if base[r['id']]['content'] is True and r['content'] is not True]
        arm['repeat_new_fact_subtraction_losses']=[r['id'] for r in arm['tasks']
            if splits[r['id']] in ('new_facts_seen_template','new_facts_new_template')
            and repeat[r['id']]['strict'] and not r['strict']]
        if name!='repeat':
            gates={'primary_transfer_exceeds_repeat':
                arm['splits']['new_facts_new_template']['strict_successes']>
                arms['repeat']['splits']['new_facts_new_template']['strict_successes'],
                'preserves_repeat_new_fact_subtraction':not arm['repeat_new_fact_subtraction_losses'],
                'preserves_baseline_exact':not arm['baseline_exact_losses'],
                'preserves_baseline_content':not arm['baseline_content_losses']}
            gates['advance']=all(gates.values());arm['gates']=gates
    result={'protocol':'research/reports/native_semantic_transfer_protocol.md',
        'tasks_sha256':digest,'sources':sources,'arms':arms}
    with Path(args.output).open('x') as destination:json.dump(result,destination,indent=2)
    print(json.dumps({name:{k:v for k,v in arm.items() if k!='tasks'} for name,arm in arms.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','evaluations','output'):parser.add_argument(f'--{name}',required=True)
    run(parser.parse_args())
