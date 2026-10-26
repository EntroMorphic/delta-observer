#!/usr/bin/env python3
"""Score all frozen retention-format endpoints with explicit gates and provenance."""

import argparse
import hashlib
import json
from pathlib import Path

from research.lfm.score_native_holdout100 import counts, score


def run(args):
    data=Path(args.data)
    raw=(data/'tasks.json').read_bytes()
    frozen=json.loads((data/'freeze.json').read_text())
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=frozen['tasks_sha256']:raise ValueError('Frozen task hash mismatch')
    manifest_bytes=(data/'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=frozen['training_manifest_sha256']:
        raise ValueError('Frozen training manifest changed')
    tasks=json.loads(raw)['tasks'];split={t['id']:t['split'] for t in tasks}
    arms={};reference=None;sources={}
    for name in ('baseline','final_only','worked_replay','extra_final'):
        path=Path(args.evaluations)/f'native_retention_format_v1_{name}/baseline.json'
        artifact=json.loads(path.read_text())
        if not artifact['complete'] or artifact['tasks_sha256']!=digest:
            raise ValueError('Incomplete or unmatched evaluation')
        settings={key:artifact[key] for key in ('model','revision','dtype','generation_config')}
        if reference is None:reference=settings
        elif reference!=settings:raise ValueError('Generation settings differ')
        if name!='baseline':
            config=artifact['adapter']['config']
            if config['arm']!=name or config['manifest_sha256']!=frozen['training_manifest_sha256']:
                raise ValueError('Adapter arm/training provenance differs')
        arm=score(tasks,artifact['results'])
        arm['splits']={group:counts([r for r in arm['tasks'] if split[r['id']]==group])
                       for group in sorted(set(split.values()))}
        arms[name]=arm
        sources[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    baseline={r['id']:r for r in arms['baseline']['tasks']}
    for name in ('final_only','worked_replay','extra_final'):
        arm=arms[name]
        arm['strict_gains']=[r['id'] for r in arm['tasks'] if r['strict'] and not baseline[r['id']]['strict']]
        arm['strict_losses']=[r['id'] for r in arm['tasks'] if not r['strict'] and baseline[r['id']]['strict']]
        arm['lost_confirmed_content']=[r['id'] for r in arm['tasks'] if r['content'] is not True and baseline[r['id']]['content'] is True]
    candidate=arms['worked_replay']
    target='new_fact_new_template'
    gates={
        'unseen_intersection_exceeds_both_controls':candidate['splits'][target]['strict_successes']>
            max(arms[n]['splits'][target]['strict_successes'] for n in ('final_only','extra_final')),
        'arithmetic_content_no_lower_than_baseline':candidate['categories']['arithmetic']['content_confirmed']>=
            arms['baseline']['categories']['arithmetic']['content_confirmed'],
        'no_baseline_content_loss':not candidate['lost_confirmed_content'],
        'no_baseline_exact_loss':not candidate['strict_losses']}
    gates['advance']=all(gates.values())
    result={'protocol':'research/reports/native_retention_format_protocol.md',
        'task_sha256':digest,'sources':sources,'arms':arms,'gates':gates}
    with Path(args.output).open('x') as destination:json.dump(result,destination,indent=2)
    print(json.dumps({'arms':{name:{k:v for k,v in arm.items() if k!='tasks'}
        for name,arm in arms.items()},'gates':gates},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','evaluations','output'):parser.add_argument(f'--{name}',required=True)
    run(parser.parse_args())
