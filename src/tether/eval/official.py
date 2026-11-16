"""Official POPE and CHAIR scoring helpers: RUCAIBox/POPE answer parsing and metrics, a check against the
upstream evaluate.py, per-image matrices and image-cluster bootstrap intervals.
"""
import ast
import collections
import contextlib
import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = ROOT / 'data/hallucination-suite'
METHODS = ['base', 'revis', 'rudder', 'grains', 'vti']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def official_choice(text):
    # Exact RUCAIBox/POPE evaluate.py algorithm, including its literal-space split.
    words = text.split('.')[0].replace(',', '').split(' ')
    return 'no' if any(word in words for word in ('No', 'not', 'no')) else 'yes'


def pope_metrics(rows, parser=official_choice):
    counts = collections.Counter((parser(r['prediction']), r['label']) for r in rows)
    tp, tn = counts[('yes', 'yes')], counts[('no', 'no')]
    fp = counts[('yes', 'no')]
    # An invalid answer to a positive question is a miss in strict sensitivity.
    fn = sum(r['label'] == 'yes' for r in rows) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'n': len(rows), 'accuracy': (tp + tn) / len(rows), 'precision': precision,
            'recall': recall, 'f1': f1, 'yes_ratio': (tp + fp) / len(rows),
            'invalid': sum(parser(r['prediction']) is None for r in rows),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def verify_upstream_pope(rows, output, method, subset, expected):
    directory = output / 'pope-official'
    directory.mkdir(exist_ok=True)
    answers = directory / f'{method}-{subset}.answers.jsonl'
    labels = DATA / 'sources' / f'coco_pope_{subset}.json'
    write_jsonl(answers, [{'question': r['metadata']['original_question'], 'answer': r['prediction']} for r in rows])
    tree = ast.parse((DATA / 'sources/pope_evaluate.py').read_text())
    replacements = {'ans_file': str(answers), 'label_file': str(labels)}
    replaced = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                node.value = ast.Constant(replacements[name])
                replaced.add(name)
    assert replaced == replacements.keys()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(ast.fix_missing_locations(tree), 'official_POPE_evaluate.py', 'exec'), {})
    text = buffer.getvalue()
    (directory / f'{method}-{subset}.metrics.txt').write_text(text)
    for label, key in [('Accuracy', 'accuracy'), ('Precision', 'precision'), ('Recall', 'recall'), ('F1 score', 'f1'), ('Yes ratio', 'yes_ratio')]:
        match = re.search(re.escape(label) + r':\s*([0-9.eE+-]+)', text)
        assert match and math.isclose(float(match.group(1)), expected[key], abs_tol=1e-12)


def matrix(rows, kind):
    by_group = collections.defaultdict(list)
    for row in rows:
        by_group[row['group_id']].append(row)
    groups = sorted(by_group)
    result = []
    for group in groups:
        values = by_group[group]
        if kind == 'pope':
            m = pope_metrics(values)
            result.append([m[k] for k in ['tp', 'tn', 'fp', 'fn']])
        else:
            assert len(values) == 1
            c = values[0]['chair']
            result.append([int(c['hallucinated_mentions'] > 0), 1, c['hallucinated_mentions'],
                           c['generated_mentions'], c['correct_distinct_objects'], c['ground_truth_objects']])
    return groups, np.asarray(result, dtype=float)


def derived(values, kind):
    if kind == 'pope':
        tp, tn, fp, fn = values.T
        return {'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'f1': 2 * tp / (2 * tp + fp + fn)}
    hs, n, hi, ni, correct, gt = values.T
    return {'CHAIR_s': hs / n, 'CHAIR_i': hi / ni, 'object_recall': correct / gt}


def intervals(records, kind):
    groups, base = matrix(records['base'], kind)
    n = len(groups)
    weights = np.random.default_rng(42).multinomial(n, np.full(n, 1 / n), size=10000)
    base_scores = derived(weights @ base, kind)
    result = {}
    for method, rows in records.items():
        method_groups, values = matrix(rows, kind)
        assert groups == method_groups
        samples = derived(weights @ values, kind)
        point = derived(values.sum(axis=0, keepdims=True), kind)
        baseline = derived(base.sum(axis=0, keepdims=True), kind)
        result[method] = {}
        for name in samples:
            assert np.isfinite(samples[name]).all()
            result[method][name] = {'value': float(point[name][0]),
                                  'ci95': np.quantile(samples[name], [.025, .975]).tolist(),
                                  'delta_vs_base': float(point[name][0] - baseline[name][0]),
                                  'delta_ci95': np.quantile(samples[name] - base_scores[name], [.025, .975]).tolist()}
    return result
