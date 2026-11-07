"""Does Mohor agree with the judge answer by answer, not only on average?

For every answer on Belebele, MKQA and XOR-TyDi, compares each scorer with an LLM judge's verdict. The judge is a
reference here, not the truth. Scorers:

- published: BERGEN's 3-gram recall against the benchmark's gold (Belebele: the gold in the question's script);
- published, other gold: the same against Belebele's gold in the other script (Belebele only);
- earlier fixes: Mohor's cleaning and short-word rule, against the benchmark's gold only;
- Mohor: the same against every written form of the gold (XOR-TyDi ships no extra forms, so there it equals the
  earlier fixes).

Per benchmark, language and question script, pooled over answer models, prints:
- AUROC of the score as a predictor of the judge's verdict;
- at a score of 0.5 or more counted as correct: precision, recall, false positives (score says correct, judge says
  wrong) and false negatives (the reverse), as shares of all answers;
- Mohor's AUROC minus the published score's.
Intervals are bootstrap 95% intervals over questions (all models' answers to a question move together).

Usage (from the repository root): python evaluate/agreement.py [--wandb RUN_NAME]
"""
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from common import log_to_wandb
from records import belebele_records, qa_records

import mohor
from mohor import best_score, published_score

THRESHOLD = 0.5
BOOTSTRAP_SAMPLES = 1000
SEED = 0


def scores(record):
    result = {'published': best_score(record['answer'], record['gold'], published_score)}
    if record['other_gold'] is not None:
        result['published, other gold'] = best_score(record['answer'], record['other_gold'], published_score)
    result['earlier fixes'] = best_score(record['answer'], record['gold'])
    result['Mohor'] = best_score(record['answer'], record['forms'])
    return result


def auroc(labels, values, weights=None):
    if labels.min() == labels.max():
        return float('nan')
    return roc_auc_score(labels, values, sample_weight=weights)


def threshold_counts(labels, values):
    predicted = values >= THRESHOLD
    true_positive = np.sum(predicted & labels)
    precision = true_positive / max(np.sum(predicted), 1)
    recall = true_positive / max(np.sum(labels), 1)
    false_positive = np.mean(predicted & ~labels)
    false_negative = np.mean(~predicted & labels)
    return precision, recall, false_positive, false_negative


def bootstrap(questions, labels, columns):
    """95% intervals of each column's AUROC, and of Mohor's AUROC minus the published score's."""
    unique_questions, index = np.unique(questions, return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = defaultdict(list)
    for _ in range(BOOTSTRAP_SAMPLES):
        weights = np.bincount(rng.integers(0, len(unique_questions), len(unique_questions)),
                              minlength=len(unique_questions))[index]
        keep = weights > 0
        values = {name: auroc(labels[keep], column[keep], weights[keep]) for name, column in columns.items()}
        for name, value in values.items():
            draws[name].append(value)
        draws['Mohor minus published'].append(values['Mohor'] - values['published'])
    return {name: (np.nanpercentile(values, 2.5), np.nanpercentile(values, 97.5)) for name, values in draws.items()}


def main():
    records = belebele_records() + qa_records()
    groups = defaultdict(list)
    for record in records:
        for judge, verdict in record['judges'].items():
            groups[(judge, record['set'], record['language'], record['question_script'])].append((record, verdict))

    rows = []
    print(mohor.SIGNATURE)
    print(f'Correct by a scorer: score >= {THRESHOLD}. FP, FN: shares of all answers. [95% bootstrap interval]')
    for key in sorted(groups):
        judge, benchmark, language, question_script = key
        pairs = groups[key]
        labels = np.array([verdict for _, verdict in pairs])
        questions = np.array([record['question'] for record, _ in pairs])
        per_record = [scores(record) for record, _ in pairs]
        columns = {name: np.array([s[name] for s in per_record]) for name in per_record[0]}
        intervals = bootstrap(questions, labels, columns)
        difference = auroc(labels, columns['Mohor']) - auroc(labels, columns['published'])
        print(f'\n{judge} judge | {benchmark} {language}, {question_script} questions | {len(pairs)} answers, '
              f'{100 * labels.mean():.1f}% judged correct | Mohor minus published AUROC {difference:+.3f} '
              f'[{intervals["Mohor minus published"][0]:+.3f}, {intervals["Mohor minus published"][1]:+.3f}]')
        print(f'  {"scorer":24}{"AUROC":>24}{"precision":>11}{"recall":>8}{"FP":>7}{"FN":>7}')
        for name, column in columns.items():
            value = auroc(labels, column)
            low, high = intervals[name]
            precision, recall, false_positive, false_negative = threshold_counts(labels, column)
            print(f'  {name:24}{value:>8.3f} [{low:.3f}, {high:.3f}]{100 * precision:>11.1f}{100 * recall:>8.1f}'
                  f'{100 * false_positive:>7.1f}{100 * false_negative:>7.1f}')
            rows.append([judge, benchmark, language, question_script, name, len(pairs), round(value, 4),
                         round(low, 4), round(high, 4), round(100 * precision, 1), round(100 * recall, 1),
                         round(100 * false_positive, 1), round(100 * false_negative, 1)])

    if '--wandb' in sys.argv:
        run_name = sys.argv[sys.argv.index('--wandb') + 1]
        log_to_wandb(run_name, ['judge', 'set', 'language', 'question script', 'scorer', 'answers', 'AUROC',
                                'AUROC low', 'AUROC high', 'precision', 'recall', 'false positives',
                                'false negatives'], rows,
                     dict(signature=mohor.SIGNATURE, threshold=THRESHOLD, bootstrap_samples=BOOTSTRAP_SAMPLES))


if __name__ == '__main__':
    main()
