"""What each part of Mohor buys: add one part at a time, on Belebele and MKQA.

Steps, each adding to the one before:
1. published: BERGEN's 3-gram recall against the benchmark's gold;
2. cleaning: both texts cleaned first (digits to 0 to 9, NFKC, invisible characters removed);
3. short-word rule: Mohor's rule for gold words under 3 letters (this is Mohor against the benchmark's gold);
4. extra gold forms: the same against every written form (Belebele: both golds; MKQA: aliases and English answer).
A fifth row, forms only, gives the extra gold forms with the published score and no other part.

For every step it also prints the random-question score: each answer scored against another question's gold
(the free credit that step gives). Belebele: romanized questions, per model. MKQA: per language and model; the
summary shows the gap to English (English minus the language, in points) as a range over the four models.

Usage (from the repository root): python evaluate/ablation.py [--wandb RUN_NAME]
"""
import sys
from collections import defaultdict

from common import another_question, log_to_wandb, percent
from records import belebele_records, qa_records

import mohor
from mohor import best_score, published_score
from mohor.clean import clean

STEPS = ['published', 'cleaning', 'short-word rule', 'extra gold forms', 'forms only']


def cleaned_published_score(answer, gold):
    return published_score(clean(answer), clean(gold))


def step_score(step, answer, record):
    if step == 'published':
        return best_score(answer, record['gold'], published_score)
    if step == 'cleaning':
        return best_score(answer, record['gold'], cleaned_published_score)
    if step == 'short-word rule':
        return best_score(answer, record['gold'])
    if step == 'extra gold forms':
        return best_score(answer, record['forms'])
    return best_score(answer, record['forms'], published_score)


def table(records):
    """(language, model) -> step -> (score, random-question score), in points."""
    by_group = defaultdict(list)
    for record in records:
        by_group[(record['language'], record['model'])].append(record)
    result = {}
    for key, group in by_group.items():
        by_question = {record['question']: record for record in group}
        other = another_question(sorted(by_question))
        result[key] = {}
        for step in STEPS:
            total = sum(step_score(step, record['answer'], record) for record in group)
            random_total = sum(step_score(step, record['answer'], by_question[other[record['question']]])
                               for record in group)
            result[key][step] = (percent(total, len(group)), percent(random_total, len(group)))
    return result


def main():
    rows = []
    print(mohor.SIGNATURE)

    belebele = table([r for r in belebele_records() if r['question_script'] == 'romanized'])
    print('\nBelebele, romanized questions: score (random-question score), out of 100')
    print(f'{"language":9}{"model":32}' + ''.join(f'{step:>20}' for step in STEPS))
    for (language, model), steps in sorted(belebele.items()):
        print(f'{language:9}{model:32}' + ''.join(f'{s:>12.1f} ({r:4.1f})' for s, r in steps.values()))
        rows += [['belebele', language, model, step, round(s, 1), round(r, 1)] for step, (s, r) in steps.items()]

    mkqa = table([r for r in qa_records() if r['set'] == 'mkqa'])
    print('\nMKQA: score (random-question score), out of 100')
    print(f'{"language":9}{"model":32}' + ''.join(f'{step:>20}' for step in STEPS))
    for (language, model), steps in sorted(mkqa.items()):
        print(f'{language:9}{model:32}' + ''.join(f'{s:>12.1f} ({r:4.1f})' for s, r in steps.values()))
        rows += [['mkqa', language, model, step, round(s, 1), round(r, 1)] for step, (s, r) in steps.items()]

    print('\nMKQA: gap to English in points, range over the four models')
    models = sorted({model for _, model in mkqa})
    languages = sorted({language for language, _ in mkqa} - {'en'})
    print(f'{"language":9}' + ''.join(f'{step:>20}' for step in STEPS))
    for language in languages:
        cells = []
        for step in STEPS:
            gaps = [mkqa[('en', m)][step][0] - mkqa[(language, m)][step][0] for m in models]
            cells.append(f'{min(gaps):>12.1f}–{max(gaps):<7.1f}')
        print(f'{language:9}' + ''.join(cells))

    if '--wandb' in sys.argv:
        run_name = sys.argv[sys.argv.index('--wandb') + 1]
        log_to_wandb(run_name, ['set', 'language', 'model', 'step', 'score', 'random-question score'], rows,
                     dict(signature=mohor.SIGNATURE, steps=STEPS))


if __name__ == '__main__':
    main()
