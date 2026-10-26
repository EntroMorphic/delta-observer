#!/usr/bin/env python3
"""Build operation-composition and paired-instruction remediation with retention."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from research.lfm.build_native_pilot_data import ngrams, normalized
from research.lfm.build_native_remediation import EN, ES, training_examples
from research.lfm.evaluate_pretrained_development import validate_suite
from research.lfm.native_lora import supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION


def examples():
    rows=[r for r in training_examples() if r['category'] in
          ('json_record','signed_sorting','spanish_number_words')]
    rng=random.Random(929)
    for i in range(128):
        a,b,c=rng.randint(3,30),rng.randint(2,12),rng.randint(1,5)
        product=a*b;result=product-c
        variants=[
            ('multiply',f'Calculate {a} * {b}. Reply with only the integer.',str(product)),
            ('subtract',f'Calculate {product} - {c}. Reply with only the integer.',str(result)),
            ('compound_final',f'A delivery contains {a} trays holding {b} cups each. Then {c} cups are removed from the delivery. How many remain? Return only the integer.',str(result)),
            ('compound_work',f'There are {a} trays of {b} cups. Remove {c} cups from the total. Show the multiplication and subtraction, then give the answer.',
             f'{a} * {b} = {product}. {product} - {c} = {result}. The answer is {result}.')]
        for mode,prompt,response in variants:
            rows.append({'category':'arithmetic','mode':mode,'prompt':prompt,'response':response})
    for i in range(40):
        names=[f'R{i+1001}',f'S{i+2001}',f'T{i+3001}']
        rng.shuffle(names);a,b,c=names
        modes={
            'chronology':f'Chronology: {c} follows {b}, which follows {a}.',
            'relations':f'{a} occurs before {b}; {c} occurs after {b}.',
            'chain':f'Temporal order from first to last: {a}, {b}, {c}.',
            'inverted':f'{b} happens after {a}. {b} happens before {c}.',
        }
        for mode,premises in modes.items():
            for earliest in (True,False):
                word=('earliest' if earliest else 'latest') if i%2==0 else ('first' if earliest else 'last')
                rows.append({'category':'transitive_ordering','mode':mode,'earliest':earliest,
                    'pair_id':f'{i}-{mode}',
                    'prompt':f'{premises} Select the {word} of those three events. Output only its identifier.',
                    'response':a if earliest else c})
    return rows


def test_suite():
    tasks=[]
    values=('agate','opal','jade','quartz','topaz','ruby','amber','onyx','garnet','zircon',
            'pearl','diamond','emerald','sapphire','amethyst','tourmaline','peridot','aquamarine','citrine','spinel')
    rng=random.Random(933)
    for i,value in enumerate(values):
        flag=bool(i%2)
        tasks.append({'id':f'record-{i:02d}','category':'json_record',
            'prompt':f'Represent the following in JSON, using exactly label and enabled as keys: the label string is "{value}"; enabled has boolean value {str(flag).lower()}. Send the JSON object alone, without Markdown.',
            'validator':'json_exact','content_validator':'json_unfenced','expected':{'label':value,'enabled':flag}})
        nums=[-51-i,39+i,-3*i-2,4+i]
        tasks.append({'id':f'sort-{i:02d}','category':'signed_sorting',
            'prompt':f'Give the ascending JSON list for these integers: {nums[1]}, {nums[2]}, {nums[3]}, {nums[0]}. Include no other text or code fences.',
            'validator':'json_exact','content_validator':'json_unfenced','expected':sorted(nums)})
        a,b,c=5+i,2+i%7,1+i%5
        tasks.append({'id':f'arithmetic-{i:02d}','category':'arithmetic',
            'prompt':f'An order consists of {a} racks with {b} plates per rack. {c} plates are taken out of the order. State only the integer number of plates left.',
            'validator':'text_exact','content_validator':'boxed_integer','expected':str(a*b-c)})
        names=[f'R{i+4101}',f'S{i+5101}',f'T{i+6101}'];rng.shuffle(names);a,b,c=names
        earliest=i%2==0
        tasks.append({'id':f'order-{i:02d}','category':'transitive_ordering',
            'prompt':f'Event {b} is later than {a} and earlier than {c}. Choose the {"earliest" if earliest else "latest"} event. Reply only with that identifier.',
            'validator':'text_exact','expected':a if earliest else c})
        tasks.append({'id':f'translation-{i:02d}','category':'spanish_number_words',
            'prompt':f'Write {EN[i]} in Spanish, as a lowercase number word. Your answer must contain only that word, including any required accent marks.',
            'validator':'text_exact','expected':ES[i]})
    return {'version':3,'role':'frozen_post_training_holdout',
        'scope':'prompt-disjoint synthetic task families; number-word facts shared with training','tasks':tasks}


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    rows=examples();suite=test_suite();validate_suite(suite)
    prior_paths=[Path('research/results/native_holdout100_frozen/tasks.json'),
        Path('research/results/native_remediation_v1_freeze/tasks.json'),
        Path('research/lfm/pretrained_development_tasks_v2.json')]
    evaluations=suite['tasks'][:]
    for path in prior_paths:evaluations.extend(json.loads(path.read_text())['tasks'])
    train_grams=[ngrams(r['prompt']) for r in rows];train_texts={normalized(r['prompt']) for r in rows}
    for task in evaluations:
        if normalized(task['prompt']) in train_texts:raise ValueError('Exact overlap')
        g=ngrams(task['prompt'])
        if any(len(g&h)/max(1,len(g|h))>=.8 for h in train_grams):raise ValueError('Near overlap')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    for row in rows:row.update(supervised_chat(tokenizer,row['prompt'],row['response'],128))
    raw=(json.dumps(rows,indent=2,ensure_ascii=False)+'\n').encode()
    (output/'remediation.json').write_bytes(raw)
    spec={'path':'remediation.json','sha256':hashlib.sha256(raw).hexdigest(),'examples':len(rows),
        'input_tokens':sum(len(r['input_ids']) for r in rows),
        'supervised_tokens':sum(r['supervised_tokens'] for r in rows),
        'category_counts':dict(Counter(r['category'] for r in rows)),
        'mode_counts':dict(Counter(r.get('mode','retention') for r in rows))}
    manifest={'model':MODEL,'revision':REVISION,'arms':{'remediation':spec},
        'scope':'explicit composition + paired wording + retention, not matched to smaller previous budgets'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    raw=(json.dumps(suite,indent=2,ensure_ascii=False)+'\n').encode();(output/'tasks.json').write_bytes(raw)
    freeze={'tasks':100,'tasks_sha256':hashlib.sha256(raw).hexdigest(),
        'training_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
        'created_before_training':True}
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2))
    print(json.dumps({'training':spec,'test':freeze},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    build(Path(parser.parse_args().output))
