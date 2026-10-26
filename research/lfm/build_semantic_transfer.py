#!/usr/bin/env python3
"""Freeze repeat/paraphrase/contrast arms at identical updates and target tokens."""

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


def render(fact,template,operation='subtract'):
    a,b,c=fact
    if operation not in ('subtract','add'):raise ValueError('Unknown operation')
    change='removed' if operation=='subtract' else 'added'
    if template==0:body=f'There are {a} trays of {b} tiles, with {c} tiles then {change}. How many tiles are there now?'
    elif template==1:body=f'Start with {a} packets holding {b} beads each. Then {"take away" if operation=="subtract" else "include"} {c} beads. Find the total left.'
    elif template==2:body=f'What is the total for {a} groups of {b} objects after {"losing" if operation=="subtract" else "gaining"} {c} objects?'
    elif template==3:body=f'Calculate ({a} * {b}) {"-" if operation=="subtract" else "+"} {c}.'
    elif template==4:body=f'A warehouse stores {a} crates with {b} cups in each crate. It subsequently {"ships out" if operation=="subtract" else "receives"} {c} cups. State the number of cups now in storage.'
    else:raise ValueError('Unknown template')
    return body+' Reply only with the decimal integer.'


def result(fact,operation):
    a,b,c=fact
    if operation not in ('subtract','add'):raise ValueError('Unknown operation')
    return str(a*b+(-c if operation=='subtract' else c))


def pools():
    candidates=[(a,b,c) for a in range(4,29) for b in range(2,10) for c in range(1,10) if a*b>c]
    random.Random(2903).shuffle(candidates)
    return candidates[:128],candidates[128:148]


def suite(train,test):
    tasks=[]
    for group,facts,template,operation in (
        ('seen_facts_new_template',train[:20],4,'subtract'),
        ('new_facts_seen_template',test,0,'subtract'),
        ('new_facts_new_template',test,4,'subtract'),
        ('counterfactual_add',test,4,'add')):
        for i,fact in enumerate(facts):
            tasks.append({'id':f'{group}-{i:02d}','category':'arithmetic','split':group,
                'operands':list(fact),'operation':operation,'prompt':render(fact,template,operation),
                'validator':'text_exact','content_validator':'boxed_integer','expected':result(fact,operation)})
    labels=('pendant','stapler','helmet','teapot','compass')
    for i,label in enumerate(labels):
        tasks.append({'id':f'record-{i}','category':'json_record','split':'retention',
            'prompt':f'Output a bare JSON object with exactly label and enabled: label must be the string "{label}" and enabled the boolean {str(i%2==0).lower()}. No code block or explanation.',
            'validator':'json_exact','content_validator':'json_unfenced','expected':{'label':label,'enabled':i%2==0}})
        values=[-82-i,37+i,-8-i,61-i]
        tasks.append({'id':f'sort-{i}','category':'signed_sorting','split':'retention',
            'prompt':f'Return only a JSON integer array sorted low to high for {values[1]}, {values[2]}, {values[3]}, {values[0]}.',
            'validator':'json_exact','content_validator':'json_unfenced','expected':sorted(values)})
        a,b,c=f'R{12101+i}',f'S{13101+i}',f'T{14101+i}'
        tasks.append({'id':f'order-{i}','category':'transitive_ordering','split':'retention',
            'prompt':f'{c} occurred after {b}. {a} occurred before {b}. Give only the identifier of the {"earliest" if i%2==0 else "latest"} occurrence.',
            'validator':'text_exact','expected':a if i%2==0 else c})
        n=3*i+1
        tasks.append({'id':f'translation-{i}','category':'spanish_number_words','split':'retention',
            'prompt':f'In Spanish, how do you write the English number word {EN[n]}? Reply with only the lowercase word, correctly accented.',
            'validator':'text_exact','expected':ES[n]})
    test_suite={'version':1,'role':'frozen_post_training_holdout','tasks':tasks,
        'scope':'80 arithmetic / 20 retention; paired templates and operations intentionally share test facts'}
    validate_suite(test_suite)
    return test_suite


def design(arm):
    if arm=='repeat':return [(0,'subtract')]*4
    if arm=='paraphrase':return [(i,'subtract') for i in range(4)]
    if arm=='contrast':return [(0,'subtract'),(1,'subtract'),(0,'add'),(1,'add')]
    raise ValueError('Unknown arm')


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    primary,test=pools();test_suite=suite(primary,test)
    excluded={' '.join(t['prompt'].casefold().split()) for t in test_suite['tasks']}
    for path in [Path('research/lfm/pretrained_development_tasks_v2.json')]+sorted(Path('research/results').glob('*freeze*/tasks.json'))+sorted(Path('research/results').glob('*frozen*/tasks.json')):
        excluded.update(' '.join(t['prompt'].casefold().split()) for t in json.loads(path.read_text())['tasks'])
    specs={}
    for arm in ('repeat','paraphrase','contrast'):
        groups=[]
        for epoch in range(4):
            for fact in primary:
                rows=[]
                for template,operation in design(arm):
                    prompt,response=render(fact,template,operation),result(fact,operation)
                    if ' '.join(prompt.casefold().split()) in excluded:raise ValueError('Exact prompt overlap')
                    row={'prompt':prompt,'response':response,'operands':list(fact),
                        'template':template,'operation':operation,
                        **supervised_chat(tokenizer,prompt,response,128)}
                    if row['supervised_tokens']!=2:raise ValueError('Target length must be exactly two positions')
                    rows.append(row)
                groups.append({'epoch':epoch,'rows':rows,'supervised_tokens':8})
        raw=(json.dumps(groups,indent=2)+'\n').encode();(output/f'{arm}.json').write_bytes(raw)
        flat=[r for g in groups for r in g['rows']]
        specs[arm]={'path':f'{arm}.json','sha256':hashlib.sha256(raw).hexdigest(),
            'updates':len(groups),'supervised_tokens':sum(r['supervised_tokens'] for r in flat),
            'sequence_exposures':len(flat),'input_tokens':sum(len(r['input_ids']) for r in flat),
            'unique_prompts':len({r['prompt'] for r in flat}),
            'operation_exposures':{op:sum(r['operation']==op for r in flat) for op in ('subtract','add')}}
    manifest={'model':MODEL,'revision':REVISION,'arms':specs,'per_update_supervised_tokens':8,
        'scope':'equal updates/target tokens/sequences; input lengths/compute and operation coverage differ',
        'primary_facts':[list(f) for f in primary],'heldout_facts':[list(f) for f in test],
        'heldout_template':4,'split_note':'seen-fact split reuses supervised facts, never prompts; test-template and add/subtract pairs share test facts intentionally',
        'caveat':'contrast changes both operation coverage and wording count; cannot attribute isolated gains to a contrastive objective; ordinary CE only'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    raw=(json.dumps(test_suite,indent=2,ensure_ascii=False)+'\n').encode();(output/'tasks.json').write_bytes(raw)
    frozen={'tasks':100,'tasks_sha256':hashlib.sha256(raw).hexdigest(),
        'training_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
        'created_before_training':True,'selection':'three fixed 512-update endpoints'}
    (output/'freeze.json').write_text(json.dumps(frozen,indent=2))
    print(json.dumps({'arms':specs,'freeze':frozen},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    build(Path(parser.parse_args().output))
