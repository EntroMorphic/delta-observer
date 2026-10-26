#!/usr/bin/env python3
"""Reproduce bounded summaries from measured semantic-gap atomic artifacts."""

import argparse
import json
import statistics
from pathlib import Path

from research.lfm.score_native_holdout100 import score,counts


def summarize(results):
    atom=json.loads((results/'native_semantic_atomics_v1/atomics.json').read_text())
    controls=json.loads((results/'native_semantic_controls_v1/controls.json').read_text())
    schema=json.loads((results/'native_schema_interference_v1/schema.json').read_text())
    if not all(x['complete'] for x in (atom,controls,schema)):raise ValueError('Incomplete audit')
    summary={'scope':'exposed diagnostic summaries; do not infer unmeasured full-model causes',
        'original_operation_pairs':{},'balanced_operation_pairs':{},'schema':{},'knockout_evaluations':{}}
    def pair_counts(pairs,balanced):
        def metrics(row,operation):return row[operation] if balanced else row['first_token_metrics'][operation]
        return {'n':len(pairs),
            'same_top1':sum(metrics(p,'subtract')['top1_id']==metrics(p,'add')['top1_id'] for p in pairs),
            'both_correct_top1':sum(metrics(p,'subtract')['target_rank']==1 and metrics(p,'add')['target_rank']==1 for p in pairs),
            'both_correct_if_only_two_answers_competed':sum(metrics(p,'subtract')['target_minus_alternative_logit']>0 and metrics(p,'add')['target_minus_alternative_logit']>0 for p in pairs),
            'mean_operation_selectivity_shift':statistics.mean(p['selectivity_shift'] if balanced else p['operation_selectivity_shift'] for p in pairs)}
    for arm in atom['arms']:
        a=atom['arms'][arm]
        summary['original_operation_pairs'][arm]=pair_counts(a['pairs'],False)
        summary['original_operation_pairs'][arm]['self_patch_max_error']=max(p['self_patch_max_logit_error'] for p in a['pairs'])
        summary['original_operation_pairs'][arm]['prefill_vs_generation_agreement']=sum(
            p['first_token_metrics'][op]['agrees_with_stored_generation_first_token'] for p in a['pairs'] for op in ('subtract','add'))
        if 'exposure' in a:summary['original_operation_pairs'][arm]['exposure']=a['exposure']
        summary['balanced_operation_pairs'][arm]=pair_counts(controls['arms'][arm]['balanced_pairs'],True)
        summary['balanced_operation_pairs'][arm]['json_content_with_eos_delay']=sum(
            r['eos_delayed_content'] is True for r in controls['arms'][arm]['retention'])
    for arm,a in schema['arms'].items():
        summary['schema'][arm]={}
        for case in a['baseline_state_patches']:
            summary['schema'][arm][case['id']]={'full_key_rank':case['full']['target_rank'],
                'baseline_patch_layers_restoring_key_top1':[p['layer'] for p in case['patches'] if p['metrics']['target_rank']==1]}
        summary['schema'][arm]['knockouts']={k['intervention']:{'content_successes':sum(r['content'] is True for r in k['cases']),
            'native_logit_max_errors':[r['native_logit_max_error'] for r in k['cases'] if 'native_logit_max_error' in r]}
            for k in a['knockouts']}
    tasks=json.loads((results/'native_semantic_transfer_v1_freeze/tasks.json').read_text())['tasks']
    native=json.loads((results/'native_semantic_transfer_v1_comparison.json').read_text())['arms']['baseline']
    base={r['id']:r for r in native['tasks']};split={t['id']:t['split'] for t in tasks}
    for arm in ('paraphrase','contrast'):
        for group in ('attention','conv'):
            path=results/f'native_semantic_{arm}_{group}_off/baseline.json'
            artifact=json.loads(path.read_text())
            if not artifact['complete'] or artifact['tasks_sha256']!=atom['tasks_sha256']:
                raise ValueError('Knockout suite mismatch')
            if artifact['adapter']['inference_ablation']['group']!=group:
                raise ValueError('Knockout group mismatch')
            a=score(tasks,artifact['results'])
            a['splits']={s:counts([r for r in a['tasks'] if split[r['id']]==s]) for s in sorted(set(split.values()))}
            a['lost_baseline_content']=[r['id'] for r in a['tasks'] if base[r['id']]['content'] is True and r['content'] is not True]
            summary['knockout_evaluations'][f'{arm}_{group}_off']=a
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('research/results'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as destination:json.dump(summarize(args.results),destination,indent=2)
