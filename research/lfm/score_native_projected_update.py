#!/usr/bin/env python3
"""Audit fixed projected endpoint, calibration/holdout distinction, and paired loss."""

import argparse
import hashlib
import json
from pathlib import Path

from research.lfm.score_native_holdout100 import counts,score


def run(args):
    results=Path(args.results);data=Path(args.data);training_path=Path(args.training)
    raw_tasks=(data/'tasks.json').read_bytes();tasks=json.loads(raw_tasks)['tasks']
    training=json.loads(training_path.read_text())
    if training['config']['manifest_sha256']!=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Training data provenance differs')
    candidate_path=results/'native_projected_update_v1_eval/baseline.json'
    candidate=json.loads(candidate_path.read_text())
    if not candidate['complete'] or candidate['tasks_sha256']!=hashlib.sha256(raw_tasks).hexdigest():
        raise ValueError('Candidate evaluation incomplete or unmatched')
    native=json.loads((results/'native_semantic_transfer_v1_baseline/baseline.json').read_text())
    original=json.loads((results/'native_semantic_transfer_v1_contrast/baseline.json').read_text())
    for artifact in (native,original):
        for key in ('tasks_sha256','model','revision','dtype','generation_config'):
            if artifact[key]!=candidate[key]:raise ValueError(f'Evaluation differs: {key}')
    if candidate['adapter']['config']!=training['config']:raise ValueError('Training/evaluation config differs')
    actual=score(tasks,candidate['results']);base=score(tasks,native['results']);old=score(tasks,original['results'])
    by_base={r['id']:r for r in base['tasks']};by_old={r['id']:r for r in old['tasks']}
    by_new={r['id']:r for r in actual['tasks']};split={t['id']:t['split'] for t in tasks}
    actual['splits']={s:counts([r for r in actual['tasks'] if split[r['id']]==s]) for s in sorted(set(split.values()))}
    actual['strict_gains_vs_original']=[r['id'] for r in actual['tasks'] if r['strict'] and not by_old[r['id']]['strict']]
    actual['strict_losses_vs_original']=[r['id'] for r in actual['tasks'] if not r['strict'] and by_old[r['id']]['strict']]
    actual['baseline_content_losses']=[r['id'] for r in actual['tasks'] if by_base[r['id']]['content'] is True and r['content'] is not True]
    actual['baseline_exact_losses']=[r['id'] for r in actual['tasks'] if by_base[r['id']]['strict'] and not r['strict']]
    actual['json_content_repairs']=[r['id'] for r in actual['tasks'] if r['category']=='json_record'
        and r['content'] is True and by_old[r['id']]['content'] is not True]
    both=sum(by_new[f'new_facts_new_template-{i:02d}']['strict'] and by_new[f'counterfactual_add-{i:02d}']['strict'] for i in range(20))
    history=training['history'];initial=history[0]['key_ce_before'];final=history[-1]['key_ce_after']
    gates={'protected_key_ce':final<=initial+.05,
        'protected_record_content':by_new['record-0']['content'] is True,
        'new_subtraction_count':actual['splits']['new_facts_new_template']['strict_successes']>=7,
        'addition_count':actual['splits']['counterfactual_add']['strict_successes']>=10,
        'both_operation_count':both>=6}
    gates['local_mechanism_pass']=all(gates.values())
    active=[h['projection'] for h in history if h['projection']['projection_active']]
    fractions=[p['removed_component_norm']/p['raw_update_norm'] for p in active]
    replay_path=results/'native_update_conflict_v1/updates.json';replay=json.loads(replay_path.read_text())
    if replay['raw_training_replication_max_adapter_error']!=0:raise ValueError('Raw replay did not replicate')
    result={'protocol':'research/reports/native_projected_update_protocol.md',
        'scope':'exposed one-key constrained diagnostic; no generalization/deployment approval',
        'calibration_case':'record-0','unprotected_json_cases':['record-1','record-2','record-3','record-4'],
        'candidate':actual,'both_operation_pairs_correct':both,'gates':gates,
        'all_baseline_content_preserved':not actual['baseline_content_losses'],
        'general_deployment_approved':False,
        'initial_protected_key_ce':initial,'final_protected_key_ce':final,
        'projection_active_steps':len(active),'removed_norm_fraction_mean_active':sum(fractions)/len(fractions),
        'removed_norm_fraction_min_active':min(fractions),'removed_norm_fraction_max_active':max(fractions),
        'raw_replay_max_adapter_error':replay['raw_training_replication_max_adapter_error'],
        'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (training_path,candidate_path,replay_path)}}
    with Path(args.output).open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='candidate'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','training','output'):parser.add_argument(f'--{name}',required=True)
    parser.add_argument('--results',default='research/results')
    run(parser.parse_args())
