#!/usr/bin/env python3
"""Fixed multi-seed retention-transfer experiment; preparation never reads test outputs."""

import argparse
import gc
import hashlib
import json
import random
import time
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.audit_native_update_conflict import assign, flatten
from research.lfm.build_native_remediation import EN, ES
from research.lfm.build_semantic_transfer import pools, result
from research.lfm.evaluate_pretrained_development import (
    chat_inputs, content_validation, run as evaluate, synchronize, validate, validate_suite,
)
from research.lfm.native_lora import adapter_state, install_lora, supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION, frozen_digest
from research.lfm.probe_lfm import choose_device
from research.lfm.score_native_holdout100 import score, counts


ARMS = ('ordinary', 'projected', 'replay', 'random')
SEEDS = (0, 1, 2)
PROTOCOL = Path('research/reports/native_retention_transfer_protocol.md')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def normalize(prompt):
    return ' '.join(prompt.casefold().split())


def direction_controls(delta, gradients, generator):
    """Same-norm nullspace and angle-matched random displacement in CPU float64."""
    if delta.ndim != 1 or not torch.isfinite(delta).all():
        raise ValueError('Invalid update')
    d = delta.double()
    basis = []
    for gradient in gradients:
        if gradient.shape != d.shape or not torch.isfinite(gradient).all():
            raise ValueError('Invalid gradient')
        vector = gradient.double().clone()
        original = float(vector.norm())
        for _ in range(2):
            for unit in basis:
                vector -= (vector @ unit) * unit
        length = float(vector.norm())
        if length > max(1e-12, original * 1e-10):
            basis.append(vector / length)
    projected = d.clone()
    for _ in range(2):
        for unit in basis:
            projected -= (projected @ unit) * unit
    norm, remaining = float(d.norm()), float(projected.norm())
    matched = projected * (norm / remaining) if remaining > norm * 1e-12 else torch.zeros_like(d)
    cosine = max(-1., min(1., float(d @ matched) / norm**2)) if norm else 1.
    noise = torch.randn(d.shape, generator=generator, dtype=torch.float64)
    if norm:
        for _ in range(2):
            noise -= (noise @ d) / norm**2 * d
    noise_norm = float(noise.norm())
    if norm and remaining > norm * 1e-12 and noise_norm > 0:
        random_delta = cosine * d + (max(0., 1 - cosine**2)**.5 * norm / noise_norm) * noise
    else:
        random_delta = matched.clone()
    return matched, random_delta, {
        'gradient_rank': len(basis), 'raw_norm': norm, 'projected_norm': float(matched.norm()),
        'random_norm': float(random_delta.norm()), 'cosine': cosine,
        'removed_fraction': float((d - projected).norm()) / norm if norm else 0.,
        'zero_projection': bool(norm and not remaining > norm * 1e-12),
        'gradient_dots_projected': [float(g.double() @ matched) for g in gradients],
    }


def schema(identifier, label, enabled, keys=('label', 'enabled'), split='protected'):
    first, second = keys
    return {'id': identifier, 'category': 'json_record', 'split': split,
            'prompt': f'Output a bare JSON object with exactly {first} and {second}: '
                      f'{first} must be the string "{label}" and {second} the boolean '
                      f'{str(enabled).lower()}. No code block or explanation.',
            'validator': 'json_exact', 'content_validator': 'json_unfenced',
            'expected': {first: label, second: enabled}}


def ordering(identifier, index, latest, alternate=False, split='protected'):
    a, b, c = f'R{index}', f'S{index+10000}', f'T{index+20000}'
    if alternate:
        prompt = (f'The sequence includes {a}, {b}, and {c}. {b} is later than {a}; '
                  f'{c} is later than {b}. Which identifier is {"last" if latest else "first"}? '
                  'Write only that identifier.')
    else:
        prompt = (f'{c} occurred after {b}. {a} occurred before {b}. Give only the '
                  f'identifier of the {"latest" if latest else "earliest"} occurrence.')
    return {'id': identifier, 'category': 'transitive_ordering', 'split': split,
            'prompt': prompt, 'validator': 'text_exact', 'expected': c if latest else a}


def build_tasks():
    old_train, old_test = pools()
    excluded = set(old_train + old_test)
    candidates = [(a, b, c) for a in range(4, 29) for b in range(2, 10)
                  for c in range(1, 10) if a*b > c and (a, b, c) not in excluded]
    random.Random(68219).shuffle(candidates)
    tasks = []
    for i, fact in enumerate(candidates[:20]):
        a, b, c = fact
        for operation in ('subtract', 'add'):
            prompt = (f'Each of {a} boxes contains {b} tokens. After inventory, the collection '
                      f'{"loses" if operation == "subtract" else "gains"} {c} tokens. '
                      'How many tokens does the collection contain now? Return only the decimal integer.')
            tasks.append({'id': f'fresh-{operation}-{i:02d}', 'category': 'arithmetic',
                          'split': operation, 'operands': list(fact), 'operation': operation,
                          'prompt': prompt, 'validator': 'text_exact',
                          'content_validator': 'boxed_integer', 'expected': result(fact, operation)})
    labels = ('sundial', 'thimble', 'badger', 'copper', 'orchid', 'ladle', 'otter',
              'marigold', 'quiver', 'goblet', 'heron', 'trowel', 'lilac', 'puffin',
              'spindle', 'dahlia', 'walrus', 'clover', 'marten', 'cobalt')
    for i, label in enumerate(labels):
        tasks.append(schema(f'fresh-record-{i:02d}', label, i % 2 == 0,
                            ('label', 'enabled') if i < 10 else ('title', 'active'),
                            'same_schema' if i < 10 else 'new_schema'))
        tasks.append(ordering(f'fresh-order-{i:02d}', 62001+i, i % 2 == 1,
                              i >= 10, 'same_order_wording' if i < 10 else 'new_order_wording'))
    for i in range(10):
        values = [-143-i, 72+i, -29-i, 114-i]
        tasks.append({'id': f'fresh-sort-{i:02d}', 'category': 'signed_sorting', 'split': 'sorting',
                      'prompt': f'Return only a JSON integer array sorted low to high for '
                                f'{values[1]}, {values[2]}, {values[3]}, {values[0]}.',
                      'validator': 'json_exact', 'content_validator': 'json_unfenced',
                      'expected': sorted(values)})
        n = (2, 3, 5, 6, 8, 9, 11, 12, 14, 15)[i]
        tasks.append({'id': f'fresh-translation-{i:02d}', 'category': 'spanish_number_words',
                      'split': 'translation', 'prompt': f'Write the Spanish word for {EN[n]}. '
                      'Use lowercase with the required accent marks and no other text.',
                      'validator': 'text_exact', 'expected': ES[n]})
    suite = {'version': 2, 'role': 'frozen_post_training_holdout', 'tasks': tasks}
    validate_suite(suite)
    return suite


def prepare(root, source, device):
    root.mkdir(parents=True, exist_ok=False)
    suite = build_tasks()
    protected = [schema('protect-record-0', 'telescope', True),
                 schema('protect-record-1', 'anvil', False),
                 ordering('protect-order-0', 51001, False),
                 ordering('protect-order-1', 51002, True)]
    source_manifest = json.loads((source/'manifest.json').read_text())
    if digest(source/'contrast.json') != source_manifest['arms']['contrast']['sha256']:
        raise ValueError('Source changed')
    groups = json.loads((source/'contrast.json').read_text())[:128]
    if len(groups) != 128 or any(g['epoch'] != 0 for g in groups):
        raise ValueError('Not exactly one complete source pass')
    historical = set()
    scanned = {}
    for path in sorted(Path('research').rglob('*.json')):
        try:
            artifact = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if isinstance(artifact, dict) and isinstance(artifact.get('tasks'), list):
            prompts = [t['prompt'] for t in artifact['tasks'] if isinstance(t, dict) and 'prompt' in t]
            if prompts:
                historical.update(map(normalize, prompts))
                scanned[str(path)] = digest(path)
    source_prompts = {normalize(r['prompt']) for g in groups for r in g['rows']}
    protected_prompts = {normalize(t['prompt']) for t in protected}
    fresh_prompts = {normalize(t['prompt']) for t in suite['tasks']}
    if fresh_prompts & (historical | source_prompts | protected_prompts):
        raise ValueError('Fresh suite overlaps known prompts')
    if protected_prompts & (historical | source_prompts):
        raise ValueError('Protection prompts overlap known tasks')
    save(root/'tasks.json', suite)
    save(root/'target.json', groups)
    save(root/'protected_tasks.json', {'tasks': protected})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               dtype=torch.float16 if device != 'cpu' else torch.float32).to(device).eval()
    generation = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.,
        max_new_tokens=192, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, use_cache=True)
    references = []
    for task in protected:
        inputs = chat_inputs(tokenizer, task['prompt']).to(device)
        with torch.inference_mode():
            ids = model.generate(**inputs, generation_config=generation)[0, inputs['input_ids'].shape[1]:].tolist()
        if not ids or ids[-1] != tokenizer.eos_token_id:
            raise ValueError('Protection reference truncated: abort, do not select replacements')
        text = tokenizer.decode(ids, skip_special_tokens=True)
        encoded = supervised_chat(tokenizer, task['prompt'], text, 512)
        if encoded['input_ids'][encoded['prompt_tokens']:] != ids:
            raise ValueError('Reference decode/re-encode mismatch')
        references.append({'id': task['id'], 'prompt': task['prompt'], 'response': text,
                           'validation': validate(task, text), 'content_validation': content_validation(task, text),
                           **encoded})
    save(root/'protected.json', references)
    save(root/'freeze.json', {'protocol_sha256': digest(PROTOCOL),
        'files': {p: digest(root/p) for p in ('tasks.json', 'target.json', 'protected_tasks.json', 'protected.json')},
        'historical_suites': scanned, 'source_sha256': digest(source/'contrast.json'),
        'seeds': list(SEEDS), 'arms': list(ARMS), 'steps': 128,
        'status': 'frozen_before_training; no heldout model outputs generated'})
    del model
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()


def verify(root):
    freeze = json.loads((root/'freeze.json').read_text())
    if freeze['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('Protocol changed')
    for name, expected in freeze['files'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Frozen data changed: {name}')
    return freeze


def train(root, arm, seed, device):
    freeze = verify(root)
    output = root/f'{arm}-seed-{seed}'
    output.mkdir(exist_ok=False)
    groups = json.loads((root/'target.json').read_text())
    references = json.loads((root/'protected.json').read_text())
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(device).eval()
    targets = install_lora(model, 8, 16)
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial = flatten(parameters)
    frozen_before = frozen_digest(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    generator = torch.Generator().manual_seed(91003+seed)
    def loss(row):
        return model(torch.tensor([row['input_ids']], device=device),
                     labels=torch.tensor([row['labels']], device=device), use_cache=False).loss
    history = []
    synchronize(device)
    start = time.perf_counter()
    for step, group in enumerate(groups, 1):
        before = flatten(parameters)
        gradients, retention_before = [], []
        for row in references:
            value = loss(row)
            if not torch.isfinite(value):
                raise RuntimeError('Nonfinite protection loss')
            values = torch.autograd.grad(value, parameters)
            gradients.append(torch.cat([g.detach().cpu().float().flatten() for g in values]))
            retention_before.append(float(value.detach()))
            del value, values
        optimizer.zero_grad(set_to_none=True)
        target_ce = 0.
        tokens = sum(r['supervised_tokens'] for r in group['rows'])
        for row in group['rows']:
            value = loss(row) * (row['supervised_tokens']/tokens)
            if not torch.isfinite(value):
                raise RuntimeError('Nonfinite target loss')
            value.backward()
            target_ce += float(value.detach())
        if arm == 'replay':
            replay_gradient = torch.stack(gradients).mean(0) * .1
            offset = 0
            for p in parameters:
                p.grad.add_(replay_gradient[offset:offset+p.numel()].view_as(p).to(p.device))
                offset += p.numel()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        raw_delta = flatten(parameters)-before
        projected, random_delta, geometry = direction_controls(raw_delta, gradients, generator)
        chosen = projected if arm == 'projected' else random_delta if arm == 'random' else raw_delta.double()
        assign(parameters, before.double()+chosen)
        with torch.no_grad():
            retention_after = [float(loss(row)) for row in references]
        realized = flatten(parameters)-before
        history.append({'step': step, 'target_ce_before': target_ce,
                        'retention_ce_before': retention_before, 'retention_ce_after': retention_after,
                        'unclipped_gradient_norm': float(norm), 'geometry': geometry,
                        'realized_norm': float(realized.double().norm()),
                        'realized_gradient_dots': [float(g.double() @ realized.double()) for g in gradients]})
        if step % 32 == 0:
            print(f'{arm} seed={seed} step={step}', flush=True)
    synchronize(device)
    elapsed = time.perf_counter()-start
    frozen_after = frozen_digest(model)
    if frozen_before != frozen_after:
        raise RuntimeError('Frozen base mutated')
    config = {'model': MODEL, 'revision': REVISION, 'rank': 8, 'alpha': 16, 'targets': targets,
              'seed': seed, 'arm': arm, 'freeze_sha256': digest(root/'freeze.json'),
              'protocol_sha256': freeze['protocol_sha256'], 'steps': len(groups)}
    torch.save({'config': config, 'adapter': adapter_state(model)}, output/'checkpoint.pt')
    save(output/'training.json', {'complete': True, 'config': config, 'history': history,
        'initial_adapter_sha256': hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
        'frozen_before_sha256': frozen_before, 'frozen_after_sha256': frozen_after,
        'runtime_seconds': elapsed, 'device': device, 'dtype': 'float32',
        'trainable_parameters': sum(p.numel() for p in parameters),
        'target_sequences': sum(len(g['rows']) for g in groups),
        'target_supervised_tokens': sum(r['supervised_tokens'] for g in groups for r in g['rows']),
        'target_input_tokens': sum(len(r['input_ids']) for g in groups for r in g['rows']),
        'retention_gradient_sequences': len(groups)*len(references),
        'retention_gradient_input_tokens': len(groups)*sum(len(r['input_ids']) for r in references),
        'retention_gradient_supervised_tokens': len(groups)*sum(r['supervised_tokens'] for r in references),
        'retention_measurement_forward_sequences': len(groups)*len(references),
        'note': 'Every arm processes identical rows/passes; wall-clock and objectives differ.'})
    del model, optimizer, parameters
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()


def evaluate_all(root):
    verify(root)
    # Refuse even the baseline until all twelve endpoints are saved.
    for seed in SEEDS:
        for arm in ARMS:
            path = root/f'{arm}-seed-{seed}'/'training.json'
            if not json.loads(path.read_text())['complete']:
                raise ValueError('Training not complete')
    destinations = [('baseline', None)] + [
        (f'{arm}-seed-{seed}', str(root/f'{arm}-seed-{seed}'/'checkpoint.pt'))
        for seed in SEEDS for arm in ARMS]
    for name, adapter in destinations:
        evaluate(Namespace(output=str(root/'eval'/name), tasks=str(root/'tasks.json'),
            max_new_tokens=192, revision=REVISION, model=MODEL, device='auto', adapter=adapter,
            disable_adapter_group=None))
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def summarize(root):
    freeze = verify(root)
    tasks = json.loads((root/'tasks.json').read_text())['tasks']
    task_by_id = {t['id']: t for t in tasks}
    names = ['baseline']+[f'{arm}-seed-{seed}' for seed in SEEDS for arm in ARMS]
    artifacts = {n: json.loads((root/'eval'/n/'baseline.json').read_text()) for n in names}
    baseline = artifacts['baseline']
    summaries = {}
    for name, artifact in artifacts.items():
        if not artifact['complete'] or artifact['tasks_sha256'] != freeze['files']['tasks.json']:
            raise ValueError('Evaluation mismatch')
        for key in ('model', 'revision', 'dtype', 'generation_config'):
            if artifact[key] != baseline[key]:
                raise ValueError('Generation settings mismatch')
        if name != 'baseline' and artifact['adapter']['sha256'] != digest(root/name/'checkpoint.pt'):
            raise ValueError('Checkpoint provenance mismatch')
        summary = score(tasks, artifact['results'])
        summary['splits'] = {s: counts([r for r in summary['tasks'] if task_by_id[r['id']]['split'] == s])
                             for s in sorted({t['split'] for t in tasks})}
        by_id = {r['id']: r for r in summary['tasks']}
        summary['both_operations_correct'] = sum(by_id[f'fresh-subtract-{i:02d}']['strict'] and
                                                by_id[f'fresh-add-{i:02d}']['strict'] for i in range(20))
        summaries[name] = summary
    def paired(candidate, reference):
        by_id = {r['id']: r for r in reference['tasks']}
        return {'strict_gains': [r['id'] for r in candidate['tasks'] if r['strict'] and not by_id[r['id']]['strict']],
                'strict_losses': [r['id'] for r in candidate['tasks'] if not r['strict'] and by_id[r['id']]['strict']],
                'content_gains': [r['id'] for r in candidate['tasks'] if r['content'] is True and by_id[r['id']]['content'] is not True],
                'content_losses': [r['id'] for r in candidate['tasks'] if r['content'] is not True and by_id[r['id']]['content'] is True]}
    gates = {}
    for seed in SEEDS:
        initial_hashes, compute = set(), []
        for arm in ARMS:
            name = f'{arm}-seed-{seed}'
            training = json.loads((root/name/'training.json').read_text())
            initial_hashes.add(training['initial_adapter_sha256'])
            compute.append(tuple(training[k] for k in ('target_sequences', 'target_supervised_tokens',
                'target_input_tokens', 'retention_gradient_sequences', 'retention_gradient_input_tokens',
                'retention_gradient_supervised_tokens', 'retention_measurement_forward_sequences')))
            summaries[name]['training'] = {k: v for k, v in training.items() if k != 'history'}
            summaries[name]['vs_native'] = paired(summaries[name], summaries['baseline'])
            summaries[name]['vs_ordinary'] = paired(summaries[name], summaries[f'ordinary-seed-{seed}'])
        if len(initial_hashes) != 1 or len(set(compute)) != 1:
            raise ValueError('Initialization or processing budget mismatch')
        def retention_losses(arm):
            return [i for i in summaries[f'{arm}-seed-{seed}']['vs_native']['content_losses']
                    if task_by_id[i]['category'] != 'arithmetic']
        losses = retention_losses('projected')
        arithmetic_losses = [i for i in summaries[f'projected-seed-{seed}']['vs_ordinary']['strict_losses']
                             if task_by_id[i]['category'] == 'arithmetic']
        gates[str(seed)] = {'retention_losses': losses, 'arithmetic_losses_vs_ordinary': arithmetic_losses,
            'retention_pass': not losses, 'arithmetic_pass': not arithmetic_losses,
            'projection_specific': len(losses) < len(retention_losses('ordinary')) and
                                   len(losses) < len(retention_losses('random'))}
    result_payload = {'protocol_sha256': digest(PROTOCOL), 'freeze_sha256': digest(root/'freeze.json'),
        'arms': summaries, 'gates_per_seed': gates,
        'all_seeds_primary_pass': all(g['retention_pass'] and g['arithmetic_pass'] for g in gates.values()),
        'all_seeds_projection_specific': all(g['projection_specific'] for g in gates.values()),
        'scope': 'Fresh locked machine-generated transfer evaluation, now exposed diagnostic; not independent blinded testing',
        'evaluation_sources': {n: digest(root/'eval'/n/'baseline.json') for n in names}}
    save(root/'comparison.json', result_payload)
    print(json.dumps({'gates': gates, 'totals': {n: {k: a[k] for k in
        ('strict_successes', 'content_confirmed', 'both_operations_correct')} for n, a in summaries.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'evaluate', 'summarize', 'all'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--device', default='auto', choices=('auto', 'mps', 'cuda', 'cpu'))
    args = parser.parse_args()
    device = choose_device(args.device)
    if args.action in ('prepare', 'all'):
        if args.source is None:
            parser.error('--source required for preparation')
        prepare(args.root, args.source, device)
    if args.action in ('train', 'all'):
        for seed in SEEDS:
            for arm in ARMS:
                train(args.root, arm, seed, device)
    if args.action in ('evaluate', 'all'):
        evaluate_all(args.root)
    if args.action in ('summarize', 'all'):
        summarize(args.root)
