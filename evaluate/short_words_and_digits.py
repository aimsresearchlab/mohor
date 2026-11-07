"""Short gold words and digit systems, computed with Mohor's rules.

Our first numbers used an older short-word fix that found a short gold word anywhere in the
answer, so "0" counted inside "940" and the Russian word "в" inside almost any Russian text. Here the fix is
Mohor's score against the benchmark's gold only: cleaning (digits, NFKC, invisible characters) plus Mohor's
short-word rule. "Net" is the gain minus the gain the same rule gives against another question's gold.

Prints:
1. Command-R's released answers (MKQA and XOR-TyDi, with and without documents): published score, Mohor's
   rules, net gain, and the language order under each.
2. Our runs, XOR-TyDi Japanese: share of answers that contain the gold word for word but score under 0.2, before
   and after; scores; the Japanese–Finnish gap.
3. Our runs, net gain per language on XOR-TyDi and MKQA.
4. Our runs, XOR-TyDi Bengali and Telugu without documents: share of answers with Bengali digits, published
   score, cleaning only, Mohor's rules.

Set BERGEN_EXPS to the folder where Chirkova et al.'s runs were unzipped (default data/bergen).
Usage (from the repository root): python evaluate/short_words_and_digits.py
"""
import ast
import glob
import json
import os
import unicodedata
from collections import defaultdict

from common import another_question, percent
from records import qa_records

import mohor
from mohor import best_score, mohor_score, published_score
from mohor.clean import clean, split_words

RUNS = (os.environ.get('BERGEN_EXPS', 'data/bergen')
        + '/beegfs/scratch/user/nchirkov/myrag/exps/bergen_langs_langprompt_release/')
LOW_SCORE = 0.2


def cleaned_published_score(answer, gold):
    return published_score(clean(answer), clean(gold))


def contains_gold(answer, golds):
    """Some gold appears word for word in the answer, after lowercasing and splitting at spaces and punctuation."""
    answer_text = ' '.join(split_words(answer))
    return any(split_words(gold) and ' '.join(split_words(gold)) in answer_text for gold in golds)


def has_native_digits(text):
    return any(unicodedata.category(c) == 'Nd' and not '0' <= c <= '9' for c in text)


def scored(answers_and_golds, score):
    """Mean best score in points, against each answer's own golds and against another question's golds."""
    keys = sorted(answers_and_golds)
    other = another_question(keys)
    own = sum(best_score(answers_and_golds[k][0], answers_and_golds[k][1], score) for k in keys)
    shuffled = sum(best_score(answers_and_golds[k][0], answers_and_golds[other[k]][1], score) for k in keys)
    return percent(own, len(keys)), percent(shuffled, len(keys))


def compare(answers_and_golds):
    published, published_random = scored(answers_and_golds, published_score)
    fixed, fixed_random = scored(answers_and_golds, mohor_score)
    return published, fixed, (fixed - published) - (fixed_random - published_random)


def command_r():
    print('\n1. Command-R, released answers: published, Mohor rules, net gain')
    for dataset in ['mkqa', 'xorqa']:
        for condition in ['langenret', 'noret']:  # with documents, without documents
            rows = []
            for folder in sorted(glob.glob(RUNS + f'{dataset}_cmdr_{condition}_*')):
                with open(folder + '/eval_dev_out.json') as f:
                    outputs = json.load(f)
                pairs = {i: (o['response'], o['label'] if isinstance(o['label'], list) else ast.literal_eval(o['label']))
                         for i, o in enumerate(outputs)}
                rows.append((folder[-2:],) + compare(pairs))
            if not rows:
                continue
            print(f'== {dataset} {"with documents" if condition == "langenret" else "without documents"}')
            for language, published, fixed, net in sorted(rows, key=lambda r: -r[1]):
                print(f'   {language}  {published:5.1f}  {fixed:5.1f}  net {net:+5.1f}')
            print('   order, published:   ', ' > '.join(r[0] for r in sorted(rows, key=lambda r: -r[1])))
            print('   order, Mohor rules: ', ' > '.join(r[0] for r in sorted(rows, key=lambda r: -r[2])))


def our_runs():
    records = qa_records()
    groups = defaultdict(dict)
    for r in records:
        groups[(r['set'], r['language'], r['model'])][r['question']] = (r['answer'], r['gold'])

    print('\n2. XOR-TyDi Japanese (with documents): answers that contain the gold word for word')
    print(f'   {"model":32}{"contain gold":>13}{"<0.2 published":>16}{"<0.2 Mohor":>12}{"published":>11}{"Mohor":>7}')
    for model in sorted({m for _, _, m in groups}):
        pairs = groups.get(('xortydi', 'ja', model))
        if not pairs:
            continue
        contained = [(a, g) for a, g in pairs.values() if contains_gold(a, g)]
        low_published = sum(best_score(a, g, published_score) < LOW_SCORE for a, g in contained)
        low_fixed = sum(best_score(a, g) < LOW_SCORE for a, g in contained)
        published, fixed, _ = compare(pairs)
        print(f'   {model:32}{len(contained):>13}{percent(low_published, len(contained)):>15.1f}%'
              f'{percent(low_fixed, len(contained)):>11.1f}%{published:>11.1f}{fixed:>7.1f}')
    print('   Japanese–Finnish gap on XOR-TyDi (Finnish minus Japanese), published -> Mohor rules:')
    for model in sorted({m for _, _, m in groups}):
        if ('xortydi', 'ja', model) in groups and ('xortydi', 'fi', model) in groups:
            ja, fi = compare(groups[('xortydi', 'ja', model)]), compare(groups[('xortydi', 'fi', model)])
            before, after = fi[0] - ja[0], fi[1] - ja[1]
            print(f'   {model:32}{before:6.1f} -> {after:5.1f} ({percent(before - after, before):.0f}% of the gap)')

    print('\n3. Net gain of Mohor rules (gain minus random-question gain), per language, in points')
    for benchmark in ['xortydi', 'mkqa', 'xortydi_noret']:
        languages = sorted({l for s, l, _ in groups if s == benchmark})
        for language in languages:
            nets = [compare(pairs)[2] for (s, l, _), pairs in groups.items() if s == benchmark and l == language]
            print(f'   {benchmark:14}{language:4}{min(nets):+5.1f} to {max(nets):+5.1f} over {len(nets)} models')

    print('\n4. XOR-TyDi without documents: Bengali digits, published, cleaning only, Mohor rules (net of random)')
    for language in ['bn', 'te']:
        for model in sorted({m for s, l, m in groups if s == 'xortydi_noret' and l == language}):
            pairs = groups[('xortydi_noret', language, model)]
            native_digits = sum(has_native_digits(a) for a, _ in pairs.values())
            published, published_random = scored(pairs, published_score)
            cleaned, cleaned_random = scored(pairs, cleaned_published_score)
            _, fixed, net = compare(pairs)
            print(f'   {language} {model:32} Bengali digits {percent(native_digits, len(pairs)):4.0f}%  '
                  f'{published:5.1f}  {cleaned:5.1f} (net {(cleaned - published) - (cleaned_random - published_random):+.1f})'
                  f'  {fixed:5.1f} (net {net:+.1f})')


def main():
    print(mohor.SIGNATURE)
    command_r()
    our_runs()


if __name__ == '__main__':
    main()
