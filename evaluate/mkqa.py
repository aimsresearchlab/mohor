"""Mohor on MKQA: each language's gap to English, next to our earlier fixes and the LLM judge.

For each model and language, prints the gap to English (English minus the language, in points) under:
- the earlier fixes alone (Mohor's scoring, against the benchmark's gold only);
- Mohor (the benchmark's gold plus MKQA's aliases and English answer);
- the Gemma 4 31B judge (percent judged correct).
It also prints Mohor's random-question score per language: each answer scored against another question's forms.

Command-R's answers are the ones Chirkova et al. released with their prompts.

Usage (from the repository root): python evaluate/mkqa.py [--wandb RUN_NAME]
"""
import gzip
import json
import sys
from collections import defaultdict

from common import another_question, log_to_wandb, percent, read_answers, read_jsonl, read_verdicts

import mohor
from mohor import best_score, mkqa_forms

PROMPTS = 'runs/inputs_2026-10-27.jsonl'
MKQA = 'data/mkqa/mkqa.jsonl.gz'
JUDGE_PROMPTS = 'runs/judge_inputs_qa_2026-10-27.jsonl'
JUDGE_VERDICTS = 'runs/judge_qa_gemma-4-31b-it-qat.jsonl'
MODELS = ['command-r', 'gpt-oss-20b', 'qwen3-30b-a3b-instruct-2507-fp8', 'mistral-small-3.2-24b-instruct']
LANGUAGES = ['ar', 'ja', 'ko', 'ru', 'th', 'zh', 'en']
COLUMNS = ['earlier fixes', 'Mohor', 'Mohor, random question', 'judge']


def main():
    prompts = {p['id']: p for p in read_jsonl(PROMPTS) if p['id'].startswith('mkqa|')}
    with gzip.open(MKQA) as f:
        mkqa_answers = {str(record['example_id']): record['answers'] for record in map(json.loads, f)}
    verdicts = read_verdicts(JUDGE_VERDICTS)
    judge_prompt_id = {(p['id'].split('|', 2)[2], p['generator']): p['id']
                       for p in read_jsonl(JUDGE_PROMPTS) if p['set'] == 'mkqa'}

    def forms(prompt_id):
        _, language, example_id = prompt_id.split('|')
        return mkqa_forms(prompts[prompt_id]['gold'], mkqa_answers[example_id], language)

    rows = []
    print(mohor.SIGNATURE)
    for model in MODELS:
        if model == 'command-r':
            answers = {prompt_id: p['cmdr_response'] for prompt_id, p in prompts.items()}
        else:
            answers = read_answers(f'runs/outputs_{model}.jsonl')
        scores = defaultdict(dict)  # language -> column -> percent
        for language in LANGUAGES:
            questions = sorted(prompt_id for prompt_id in prompts if prompt_id.split('|')[1] == language)
            other = another_question(questions)
            totals = dict.fromkeys(COLUMNS, 0.0)
            for prompt_id in questions:
                answer = answers[prompt_id]
                totals['earlier fixes'] += best_score(answer, prompts[prompt_id]['gold'])
                totals['Mohor'] += best_score(answer, forms(prompt_id))
                totals['Mohor, random question'] += best_score(answer, forms(other[prompt_id]))
                totals['judge'] += verdicts.get(judge_prompt_id.get((prompt_id, model)), False)
            for column in COLUMNS:
                scores[language][column] = percent(totals[column], len(questions))

        print(f'\n{model}: gap to English in points (random question: Mohor score against another question)')
        print(f'{"language":10}' + ''.join(f'{column:>24}' for column in COLUMNS))
        for language in LANGUAGES:
            gaps = [scores['en'][column] - scores[language][column] for column in ['earlier fixes', 'Mohor']]
            judge_gap = scores['en']['judge'] - scores[language]['judge']
            random_score = scores[language]['Mohor, random question']
            values = [round(value, 1) for value in gaps + [random_score, judge_gap]]
            rows.append([model, language] + values)
            print(f'{language:10}' + ''.join(f'{value:>24.1f}' for value in values))

    if '--wandb' in sys.argv:
        run_name = sys.argv[sys.argv.index('--wandb') + 1]
        log_to_wandb(run_name, ['model', 'language', 'gap: earlier fixes', 'gap: Mohor', 'Mohor, random question',
                                'gap: judge'], rows, dict(signature=mohor.SIGNATURE, set='mkqa', judge=JUDGE_VERDICTS))


if __name__ == '__main__':
    main()
