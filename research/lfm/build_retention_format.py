#!/usr/bin/env python3
"""Build update/token-matched answer-format and worked-replay controls."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from research.lfm.build_native_remediation import EN, ES
from research.lfm.evaluate_pretrained_development import validate_suite
from research.lfm.native_lora import supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION


def text(fact, template, worked=False):
    a,b,c=fact
    if template==0:
        body=f'A shipment has {a} cartons with {b} tiles each. After {c} tiles are removed, how many tiles remain?'
    elif template==1:
        body=f'Compute ({a} * {b}) - {c}.'
    elif template==2:
        body=f'{c} screws are taken away from a supply consisting of {a} packets of {b} screws. Find the remaining quantity.'
    else:
        raise ValueError('Unknown template')
    return body+(' Show the multiplication and subtraction and give the final answer.' if worked
                 else ' Return only the decimal integer, with no explanation.')


def target(fact, worked=False):
    a,b,c=fact
    return f'{a} * {b} = {a*b}. {a*b} - {c} = {a*b-c}. The answer is {a*b-c}.' if worked else str(a*b-c)


def pack(pool, budget, start):
    """Exact supervised-token packing; no fake/padded supervised positions."""
    if budget<0:raise ValueError('Negative remaining budget')
    ordered=pool[start % len(pool):]+pool[:start % len(pool)]
    paths={0:[]}
    for row in ordered:
        # 0/1 subset sum: do not satisfy a budget by repeating one row many times.
        for total in sorted(list(paths),reverse=True):
            new=total+row['supervised_tokens']
            if new<=budget and new not in paths:
                paths[new]=paths[total]+[row]
    if budget not in paths:raise ValueError(f'Cannot pack exact token budget {budget}')
    return paths[budget]


def fact_pools():
    rng=random.Random(1847)
    all_facts=[(a,b,c) for a in range(4,29) for b in range(2,10) for c in range(1,10) if a*b>c]
    rng.shuffle(all_facts)
    return all_facts[:128],all_facts[128:256],all_facts[256:296]


def new_suite(primary, extra, heldout):
    tasks=[]
    for split,facts,template in [('seen_fact_new_template',primary[:20],2),
                                 ('new_fact_seen_template',heldout[:20],0),
                                 ('new_fact_new_template',heldout[20:40],2)]:
        for i,fact in enumerate(facts):
            tasks.append({'id':f'{split}-{i:02d}','category':'arithmetic','split':split,
                'operands':list(fact),'prompt':text(fact,template),
                'validator':'text_exact','content_validator':'boxed_integer','expected':target(fact)})
    words=('sandal','bucket','candle','pillow','button','wallet','bottle','ladder','sponge','towel')
    for i,word in enumerate(words):
        tasks.append({'id':f'record-{i}','category':'json_record','split':'retention',
            'prompt':f'Provide only JSON with exactly two fields: label is "{word}" and enabled is {str(i%2==0).lower()}. Do not include any Markdown.',
            'validator':'json_exact','content_validator':'json_unfenced','expected':{'label':word,'enabled':i%2==0}})
        vals=[-71-i,12+i,-2-i,91-i]
        tasks.append({'id':f'sort-{i}','category':'signed_sorting','split':'retention',
            'prompt':f'Sort {vals[3]}, {vals[0]}, {vals[2]}, {vals[1]} ascending. Reply solely with a JSON integer array.',
            'validator':'json_exact','content_validator':'json_unfenced','expected':sorted(vals)})
        a,b,c=f'R{8101+i}',f'S{9101+i}',f'T{10101+i}'
        tasks.append({'id':f'order-{i}','category':'transitive_ordering','split':'retention',
            'prompt':f'{a} happened before {b}, and {b} happened before {c}. State only the ID of the {"first" if i%2 else "last"} event.',
            'validator':'text_exact','expected':a if i%2 else c})
        n=2*i
        tasks.append({'id':f'translation-{i}','category':'spanish_number_words','split':'retention',
            'prompt':f'Translate the number word "{EN[n]}" to Spanish. Answer only with its lowercase Spanish word.',
            'validator':'text_exact','expected':ES[n]})
    suite={'version':1,'role':'frozen_post_training_holdout','tasks':tasks,
        'scope':'60 arithmetic across registered fact/template splits and 40 retention; seen-fact split intentionally reuses supervised facts'}
    validate_suite(suite)
    train_facts=set(primary)|set(extra)
    assert not train_facts & set(heldout)
    return suite


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    primary,extra,heldout=fact_pools()
    def row(fact,template=0,worked=False):
        prompt,response=text(fact,template,worked),target(fact,worked)
        return {'operands':list(fact),'worked':worked,'template':template,
                'prompt':prompt,'response':response,**supervised_chat(tokenizer,prompt,response,128)}
    # Final-answer pools include both seen training forms; held-out template 2 is excluded.
    finals=[row(fact,i%2) for i,fact in enumerate(primary)]
    new_finals=[row(fact,i%2) for i,fact in enumerate(extra)]
    worked=[row(fact,i%2,True) for i,fact in enumerate(primary)]
    suite=new_suite(primary,extra,heldout)
    excluded={' '.join(t['prompt'].casefold().split()) for t in suite['tasks']}
    for filename in ('research/results/native_rule_v2_freeze/tasks.json',
                     'research/results/native_remediation_v1_freeze/tasks.json',
                     'research/results/native_holdout100_frozen/tasks.json',
                     'research/lfm/pretrained_development_tasks_v2.json'):
        excluded.update(' '.join(t['prompt'].casefold().split()) for t in json.loads(Path(filename).read_text())['tasks'])
    if any(' '.join(r['prompt'].casefold().split()) in excluded for r in finals+new_finals+worked):
        raise ValueError('Prompt overlap with evaluation')
    specs={}
    for arm in ('final_only','worked_replay','extra_final'):
        groups=[]
        for i,main in enumerate(finals):
            aux=[worked[i]] if arm=='worked_replay' else []
            used=main['supervised_tokens']+sum(r['supervised_tokens'] for r in aux)
            pool=new_finals if arm=='extra_final' else [r for r in finals if r['prompt']!=main['prompt']]
            filler=pack(pool,48-used,i)
            groups.append({'rows':[main]+aux+filler,'supervised_tokens':48})
        data=(json.dumps(groups,indent=2,ensure_ascii=False)+'\n').encode()
        (output/f'{arm}.json').write_bytes(data)
        flat=[r for group in groups for r in group['rows']]
        specs[arm]={'path':f'{arm}.json','sha256':hashlib.sha256(data).hexdigest(),
            'updates':len(groups),'supervised_tokens':sum(r['supervised_tokens'] for r in flat),
            'input_tokens':sum(len(r['input_ids']) for r in flat),'sequence_exposures':len(flat),
            'unique_prompts':len({r['prompt'] for r in flat}),
            'worked_sequence_exposures':sum(r['worked'] for r in flat)}
    manifest={'model':MODEL,'revision':REVISION,'arms':specs,
        'per_update_supervised_tokens':48,'scope':'token/update matched, not input/compute matched',
        'primary_facts':[list(f) for f in primary],'extra_facts':[list(f) for f in extra],
        'heldout_facts':[list(f) for f in heldout],'training_templates':[0,1],'heldout_templates':[2],
        'coverage_note':'extra control has new facts; worked replay reuses primary facts and includes long traces; effects cannot be attributed to native replay KL',
        'split_note':'seen-fact/new-template reuses training arithmetic intentionally; other arithmetic splits have zero operand-triple overlap'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    data=(json.dumps(suite,indent=2,ensure_ascii=False)+'\n').encode()
    (output/'tasks.json').write_bytes(data)
    freeze={'tasks':100,'tasks_sha256':hashlib.sha256(data).hexdigest(),
        'training_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
        'created_before_training':True,'selection':'all three fixed 128-update endpoints'}
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2))
    print(json.dumps({'arms':specs,'freeze':freeze},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    build(Path(parser.parse_args().output))
