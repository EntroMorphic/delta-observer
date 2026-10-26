#!/usr/bin/env python3
"""Build coverage-audited remediation data and freeze a new prompt-disjoint test."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from research.lfm.build_native_pilot_data import ngrams, normalized
from research.lfm.evaluate_pretrained_development import validate_suite
from research.lfm.native_lora import supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION


WORDS = ("lantern", "mug", "notebook", "ribbon", "basket", "kettle", "scarf", "clock",
         "copper", "silver", "marble", "granite", "velvet", "cotton", "leather", "linen")
EN = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
      "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
ES = ("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
      "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve")


def training_examples():
    rng = random.Random(726)
    rows = []
    for i in range(128):
        word, flag = WORDS[i % 16], bool((i // 16) % 2)
        style = ("compact", "spaced", "pretty")[i % 3]
        prompt = (f'Convert this entry to a JSON object: label equals "{word}" and enabled equals '
                  f'{str(flag).lower()}. Keep exactly those fields. Use {style} JSON without a Markdown fence.')
        expected = {"label": word, "enabled": flag}
        response = json.dumps(expected, indent=2) if style == 'pretty' else json.dumps(
            expected, separators=(',', ':') if style == 'compact' else (', ', ': '))
        rows.append({"category": "json_record", "style": style, "value": word,
                     "prompt": prompt, "response": response})
        values = rng.sample(range(-80, 81), 4)
        rows.append({"category": "signed_sorting",
            "prompt": f"Arrange {', '.join(map(str, values))} from least to greatest. Your answer must be only a JSON list of integers; do not use code fences.",
            "response": json.dumps(sorted(values), separators=(',', ':'))})
        a, b, c = rng.randint(3, 25), rng.randint(2, 12), rng.randint(1, 5)
        rows.append({"category": "arithmetic",
            "prompt": f"{a} packages each hold {b} screws. Remove {c} screws from the total. Answer with the remaining count as an integer only.",
            "response": str(a * b - c)})
        a, b, c = f'R{i+501}', f'S{i+701}', f'T{i+901}'
        earliest = i % 2 == 0
        rows.append({"category": "transitive_ordering",
            "prompt": f"{a} occurs before {b}; {c} occurs after {b}. Give only the identifier of the {'first' if earliest else 'last'} event.",
            "response": a if earliest else c})
        n = i % 20
        rows.append({"category": "spanish_number_words",
            "prompt": f"English number: {EN[n]}. Write its Spanish translation as one lowercase word with accents if needed. Nothing else.",
            "response": ES[n]})
    return rows


def holdout():
    words = ("aster", "dahlia", "tulip", "orchid", "poppy", "violet", "lilac", "iris", "rose", "lily",
             "sage", "thyme", "basil", "mint", "parsley", "fennel", "cumin", "clove", "nutmeg", "saffron")
    tasks = []
    for i, word in enumerate(words):
        flag = i % 2 == 0
        tasks.append({"id": f"record-{i:02d}", "category": "json_record",
            "prompt": f'Produce a JSON record for a switch named "{word}". Its text goes under label, and its enabled boolean is {str(flag).lower()}. The record must have only these two keys. Do not wrap it in Markdown or prose.',
            "validator": "json_exact", "content_validator": "json_unfenced",
            "expected": {"label": word, "enabled": flag}})
        values = [-(i+43), i+17, 75-i, -2*i-7]
        tasks.append({"id": f"sort-{i:02d}", "category": "signed_sorting",
            "prompt": f"Increasing-order exercise: {values[2]}; {values[0]}; {values[3]}; {values[1]}. Reply with the sorted integers in a JSON array and no surrounding explanation or Markdown.",
            "validator": "json_exact", "content_validator": "json_unfenced", "expected": sorted(values)})
        a, b, c = 9+i, 3+i%8, 6+i
        tasks.append({"id": f"arithmetic-{i:02d}", "category": "arithmetic",
            "prompt": f"A delivery brings {a} trays, with {b} cups per tray. After {c} cups are taken away, what is the cup count left? Respond solely with a decimal integer.",
            "validator": "text_exact", "content_validator": "boxed_integer", "expected": str(a*b-c)})
        a,b,c=f'R{i+201}',f'S{i+301}',f'T{i+401}'
        earliest=i%2==0
        tasks.append({"id": f"order-{i:02d}", "category": "transitive_ordering",
            "prompt": f"Chronology: {c} follows {b}, which follows {a}. Select the {'earliest' if earliest else 'latest'} of the three. Output only its identifier.",
            "validator": "text_exact", "expected": a if earliest else c})
        tasks.append({"id": f"translation-{i:02d}", "category": "spanish_number_words",
            "prompt": f"For a Spanish glossary, replace the English numeral word {EN[i]} with its Spanish equivalent. Your complete response should be the lowercase Spanish word, correctly accented, with no other characters.",
            "validator": "text_exact", "expected": ES[i]})
    return {"version": 2, "role": "frozen_post_training_holdout",
        "scope": "new prompts; Spanish number mappings overlap training knowledge; task families deliberately shared",
        "tasks": tasks}


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    rows, suite = training_examples(), holdout()
    validate_suite(suite)
    prior = json.loads(Path('research/results/native_holdout100_frozen/tasks.json').read_text())['tasks']
    development = json.loads(Path('research/lfm/pretrained_development_tasks_v2.json').read_text())['tasks']
    train_texts = {normalized(row['prompt']) for row in rows}
    evaluation = suite['tasks'] + prior + development
    max_similarity = 0.
    for task in evaluation:
        grams = ngrams(task['prompt'])
        if normalized(task['prompt']) in train_texts:
            raise ValueError('Exact training/evaluation overlap')
        for row in rows:
            other = ngrams(row['prompt'])
            similarity = len(grams & other) / max(1, len(grams | other))
            max_similarity = max(max_similarity, similarity)
            if similarity >= .8:
                raise ValueError('Near-duplicate training/evaluation overlap')
    for row in rows:
        row.update(supervised_chat(tokenizer, row['prompt'], row['response'], 128))
    categories = dict(Counter(row['category'] for row in rows))
    values = dict(Counter(row['value'] for row in rows if row['category']=='json_record'))
    styles = dict(Counter(row['style'] for row in rows if row['category']=='json_record'))
    if set(values) != set(WORDS) or set(values.values()) != {8} or set(categories.values()) != {128}:
        raise ValueError('Coverage gate failed')
    raw=(json.dumps(rows,indent=2,ensure_ascii=False)+'\n').encode()
    (output/'remediation.json').write_bytes(raw)
    test_bytes=(json.dumps(suite,indent=2,ensure_ascii=False)+'\n').encode()
    (output/'tasks.json').write_bytes(test_bytes)
    spec={'path':'remediation.json','sha256':hashlib.sha256(raw).hexdigest(),'examples':len(rows),
          'supervised_tokens':sum(row['supervised_tokens'] for row in rows),
          'input_tokens':sum(len(row['input_ids']) for row in rows),
          'category_counts':categories,'copied_value_counts':values,'style_counts':styles}
    manifest={'model':MODEL,'revision':REVISION,'arms':{'remediation':spec},
        'scope':'remediation bundle, not a budget-matched causal comparison to the old pilot',
        'max_evaluation_train_word5_jaccard':max_similarity,
        'exclusion_caveat':'Prompt-disjoint lexical screen; shared task families and Spanish facts, not pretraining decontamination'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    freeze={'tasks':100,'tasks_sha256':hashlib.sha256(test_bytes).hexdigest(),
            'training_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
            'created_before_training':True,'selection':'fixed final endpoint, no intermediate test evaluations'}
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2))
    print(json.dumps({'training':spec,'test':freeze},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    build(Path(parser.parse_args().output))
