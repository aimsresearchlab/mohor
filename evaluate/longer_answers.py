"""Belebele answers made again with a 1,024-token limit, next to the first answers (128 tokens).

A third to a half of Sonnet's and Qwen's first Belebele answers stopped at 128 tokens. Both models answered every
Belebele prompt again with a 1,024-token limit. This script prints, per language, question script and
run: the share of answers cut off, the published score against the gold in the question's script and in the other
script, Mohor, Mohor against another question's golds, and the share judged correct by Gemma 4 31B and Seed-OSS-36B
(romanized questions: both golds shown). Then the model order on romanized questions with the longer answers in
place of the first ones. A run or judge without files yet is left out.

Usage (from the repository root): python evaluate/longer_answers.py [--wandb RUN_NAME]
"""
import os
import sys
from collections import defaultdict

from common import another_question, log_to_wandb, percent
from records import belebele_records

import mohor
from mohor import best_score, published_score

PAIRS = [('claude-sonnet-5-5', 'claude-sonnet-5-5-1024'),
         ('qwen3-30b-a3b-instruct-2507-fp8', 'qwen3-30b-a3b-instruct-2507-fp8-1024')]
OTHERS = ['mistral-small-3.2-24b-instruct', 'gpt-oss-20b']
SHORT_NAME = {'claude-sonnet-5-5': 'Sonnet 128', 'claude-sonnet-5-5-1024': 'Sonnet 1024',
              'qwen3-30b-a3b-instruct-2507-fp8': 'Qwen 128', 'qwen3-30b-a3b-instruct-2507-fp8-1024': 'Qwen 1024',
              'mistral-small-3.2-24b-instruct': 'Mistral', 'gpt-oss-20b': 'gpt-oss'}
COLUMNS = ['cut off', 'own-script gold', 'other-script gold', 'Mohor', 'Mohor, random question', 'Gemma', 'Seed']


def column_values(record, other):
    return {'cut off': record['cut_off'],
            'own-script gold': best_score(record['answer'], record['gold'], published_score),
            'other-script gold': best_score(record['answer'], record['other_gold'], published_score),
            'Mohor': best_score(record['answer'], record['forms']),
            'Mohor, random question': best_score(record['answer'], other['forms']),
            'Gemma': record['judges'].get('gemma'),
            'Seed': record['judges'].get('seed')}


def main():
    models = [m for pair in PAIRS for m in pair if os.path.exists(f'runs/outputs_{m}.jsonl')] + OTHERS
    records = belebele_records(models)
    by_key = {(r['language'], r['question_script'], r['model'], r['question']): r for r in records}
    rows = []
    print(mohor.SIGNATURE)
    for language in ['bn', 'ne']:
        for script in ['romanized', 'native']:
            questions = sorted({r['question'] for r in records if r['language'] == language})
            other = another_question(questions)
            print(f'\n{language}, {script} questions ({len(questions)})')
            print(f'{"run":13}' + ''.join(f'{column:>24}' for column in COLUMNS))
            table = defaultdict(dict)
            for model in models:
                values = [column_values(by_key[(language, script, model, q)], by_key[(language, script, model, other[q])])
                          for q in questions]
                for column in COLUMNS:
                    judged = [v[column] for v in values if v[column] is not None]
                    table[column][model] = percent(sum(judged), len(judged)) if judged else None
                cells = ''.join(f'{"-":>24}' if table[c][model] is None else f'{table[c][model]:>24.1f}' for c in COLUMNS)
                print(f'{SHORT_NAME[model]:13}' + cells)
                rows.append([language, script, model] +
                            [None if table[c][model] is None else round(table[c][model], 1) for c in COLUMNS])
            if script == 'romanized':
                for label, keep in [('first answers', [p[0] for p in PAIRS]), ('longer answers', [p[-1] for p in PAIRS])]:
                    shown = [m for m in keep if m in models] + OTHERS
                    for column in ['own-script gold', 'other-script gold', 'Mohor', 'Gemma', 'Seed']:
                        ranked = sorted((m for m in shown if table[column][m] is not None), key=lambda m: -table[column][m])
                        if len(ranked) == len(shown):
                            print(f'  {label}, order by {column}: ' + ' > '.join(SHORT_NAME[m] for m in ranked))

    if '--wandb' in sys.argv:
        run_name = sys.argv[sys.argv.index('--wandb') + 1]
        log_to_wandb(run_name, ['language', 'question script', 'model'] + COLUMNS, rows,
                     dict(signature=mohor.SIGNATURE, set='belebele', pairs=PAIRS))


if __name__ == '__main__':
    main()
