"""Belebele's romanized questions without the two known confounds.

Our prompts omit the four answer choices, so questions that refer to them ("which of the following") are hard to
answer; and many answers stopped at the 128-token limit. This script repeats the Belebele table on:

1. all 900 questions;
2. the 621 questions whose English version does not mention "the following" (the proposed main set);
3. finished answers only, all questions (each model keeps a different set of questions);
4. the 621 questions, keeping only questions that every model finished (one shared set).

Columns: the published score against the romanized gold and against the native-script gold, Mohor (both golds),
Mohor against another question's golds, and the share judged correct by Gemma 4 31B and by Claude Haiku 5.5 (both
golds shown; Haiku does not judge Sonnet). After each table, the model order under each column.

Usage (from the repository root): python evaluate/belebele_subsets.py [--wandb RUN_NAME]
"""
import sys
from collections import defaultdict

from common import another_question, log_to_wandb, percent
from records import BELEBELE_MODELS, belebele_records

import mohor
from mohor import best_score, published_score

COLUMNS = ['romanized gold', 'native gold', 'Mohor', 'Mohor, random question', 'Gemma', 'Haiku']
SHORT_NAME = {'gpt-oss-20b': 'gpt-oss', 'qwen3-30b-a3b-instruct-2507-fp8': 'Qwen',
              'mistral-small-3.2-24b-instruct': 'Mistral', 'claude-sonnet-5-5': 'Sonnet'}


def column_values(record, other):
    return {'romanized gold': best_score(record['answer'], record['gold'], published_score),
            'native gold': best_score(record['answer'], record['other_gold'], published_score),
            'Mohor': best_score(record['answer'], record['forms']),
            'Mohor, random question': best_score(record['answer'], other['forms']),
            'Gemma': record['judges'].get('gemma'),
            'Haiku': record['judges'].get('haiku')}


def main():
    records = [r for r in belebele_records() if r['question_script'] == 'romanized']
    by_key = {(r['language'], r['model'], r['question']): r for r in records}
    rows = []
    print(mohor.SIGNATURE)
    for language in ['bn', 'ne']:
        questions = sorted({r['question'] for r in records if r['language'] == language})
        other = another_question(questions)
        values = {(model, q): column_values(by_key[(language, model, q)], by_key[(language, model, other[q])])
                  for model in BELEBELE_MODELS for q in questions}
        main_set = [q for q in questions if not by_key[(language, BELEBELE_MODELS[0], q)]['refers_to_choices']]
        finished_by_all = [q for q in main_set
                           if not any(by_key[(language, model, q)]['cut_off'] for model in BELEBELE_MODELS)]
        subsets = [
            ('all questions', lambda model: questions),
            ('without "the following"', lambda model: main_set),
            ('finished answers only', lambda model: [q for q in questions if not by_key[(language, model, q)]['cut_off']]),
            ('without "the following", finished by every model', lambda model: finished_by_all)]
        for subset_name, subset in subsets:
            print(f'\n{language}: {subset_name}')
            print(f'{"model":10}{"questions":>10}' + ''.join(f'{column:>24}' for column in COLUMNS))
            table = defaultdict(dict)
            for model in BELEBELE_MODELS:
                kept = subset(model)
                for column in COLUMNS:
                    judged = [values[(model, q)][column] for q in kept if values[(model, q)][column] is not None]
                    table[column][model] = percent(sum(judged), len(judged)) if judged else None
                cells = ''.join(f'{"-":>24}' if table[c][model] is None else f'{table[c][model]:>24.1f}' for c in COLUMNS)
                print(f'{SHORT_NAME[model]:10}{len(kept):>10}' + cells)
                rows.append([language, subset_name, model, len(kept)] +
                            [None if table[c][model] is None else round(table[c][model], 1) for c in COLUMNS])
            for column in ['romanized gold', 'native gold', 'Mohor', 'Gemma', 'Haiku']:
                ranked = sorted((m for m in BELEBELE_MODELS if table[column][m] is not None),
                                key=lambda m: -table[column][m])
                print(f'  order by {column}: ' + ' > '.join(SHORT_NAME[m] for m in ranked))

    if '--wandb' in sys.argv:
        run_name = sys.argv[sys.argv.index('--wandb') + 1]
        log_to_wandb(run_name, ['language', 'subset', 'model', 'questions'] + COLUMNS, rows,
                     dict(signature=mohor.SIGNATURE, set='belebele'))


if __name__ == '__main__':
    main()
