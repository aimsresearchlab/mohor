"""Mohor on Belebele's romanized questions, next to the single-gold scores and the LLM judge.

For each language and model, prints the share of answers to romanized questions that are:
- scored against the romanized gold only (published score);
- scored against the native-script gold only (published score);
- scored by Mohor against both golds;
- scored by Mohor against another question's golds (the random-question check: free credit);
- judged correct by the Gemma 4 31B judge, shown both golds.

Usage (from the repository root): python evaluate/belebele.py [--wandb RUN_NAME]
"""
import sys

from common import another_question, log_to_wandb, percent, read_answers, read_jsonl, read_verdicts

import mohor
from mohor import belebele_forms, best_score, published_score

PROMPTS = 'runs/inputs_2026-10-27.jsonl'
JUDGE_VERDICTS = 'runs/judge_gemma-4-31b-it-qat.jsonl'
MODELS = ['claude-sonnet-5-5', 'mistral-small-3.2-24b-instruct', 'gpt-oss-20b', 'qwen3-30b-a3b-instruct-2507-fp8']
LANGUAGES = [('bn', 'ben_Latn', 'ben_Beng'), ('ne', 'npi_Latn', 'npi_Deva')]  # language, romanized, native
COLUMNS = ['romanized gold', 'native gold', 'Mohor', 'Mohor, random question', 'judge']


def main():
    gold = {p['id']: p['gold'] for p in read_jsonl(PROMPTS) if p['id'].startswith('belebele|')}
    verdicts = read_verdicts(JUDGE_VERDICTS)
    rows = []
    print(mohor.SIGNATURE)
    for language, romanized, native in LANGUAGES:
        items = sorted(prompt_id.split('|', 2)[2] for prompt_id in gold if prompt_id.startswith(f'belebele|{romanized}|'))
        other = another_question(items)

        def forms(item):
            return belebele_forms(gold[f'belebele|{romanized}|{item}'], gold[f'belebele|{native}|{item}'])

        print(f'\n{language}: {len(items)} romanized questions')
        print(f'{"model":32}' + ''.join(f'{column:>24}' for column in COLUMNS))
        for model in MODELS:
            answers = read_answers(f'runs/outputs_{model}.jsonl')
            totals = dict.fromkeys(COLUMNS, 0.0)
            for item in items:
                answer = answers[f'belebele|{romanized}|{item}']
                totals['romanized gold'] += best_score(answer, gold[f'belebele|{romanized}|{item}'], published_score)
                totals['native gold'] += best_score(answer, gold[f'belebele|{native}|{item}'], published_score)
                totals['Mohor'] += best_score(answer, forms(item))
                totals['Mohor, random question'] += best_score(answer, forms(other[item]))
                totals['judge'] += verdicts.get(f'answer|{model}|{language}|{item}|q=romanized|gold=both', False)
            values = [round(percent(totals[column], len(items)), 1) for column in COLUMNS]
            rows.append([language, model] + values)
            print(f'{model:32}' + ''.join(f'{value:>24.1f}' for value in values))

    if '--wandb' in sys.argv:
        run_name = sys.argv[sys.argv.index('--wandb') + 1]
        log_to_wandb(run_name, ['language', 'model'] + COLUMNS, rows,
                     dict(signature=mohor.SIGNATURE, set='belebele', judge=JUDGE_VERDICTS))


if __name__ == '__main__':
    main()
