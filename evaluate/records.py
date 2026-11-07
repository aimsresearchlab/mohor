"""Every answer we scored, one record each, with its gold forms, judge verdicts and flags.

Used by the answer-by-answer scripts (agreement.py, ablation.py, belebele_subsets.py). Each record is a dict:

- set: 'belebele', 'mkqa', 'xortydi' or 'xortydi_noret'
- language: as in the prompt ids ('th', 'ja', ...); for Belebele 'bn' or 'ne'
- question_script: 'native' or 'romanized' (Belebele); 'native' elsewhere
- model, question (an id shared by every model's answer to the same question), answer
- gold: the benchmark's gold (Belebele: the gold in the question's own script)
- forms: every written form Mohor uses (Belebele: both golds; MKQA: plus aliases and the English answer;
  XOR-TyDi: the benchmark's gold only, since it ships nothing more)
- other_gold: the gold in the other script (Belebele only, else None)
- judges: {'gemma': True/False, 'haiku': True/False} where that judge gave a verdict
- cut_off: the answer stopped at the token limit
- refers_to_choices: Belebele questions whose English version mentions "the following" (our prompts omit the
  four choices)

Run the scripts from the repository root.
"""
import glob
import gzip
import json
import re

from common import read_answers, read_jsonl, read_verdicts

from mohor import belebele_forms, mkqa_forms

PROMPTS = 'runs/inputs_2026-10-27.jsonl'
MKQA = 'data/mkqa/mkqa.jsonl.gz'
BELEBELE_ENGLISH = 'data/belebele/eng_Latn.jsonl'
# Each judge's verdict files (patterns); a file that does not exist yet is skipped. The -1024 files judge the answers
# made again with a 1,024-token limit; the Seed-OSS file holds every prompt.
JUDGES = {'gemma': ['runs/judge_gemma-4-31b-it-qat.jsonl', 'runs/judge_*-1024_gemma-4-31b-it-qat.jsonl'],
          'haiku': ['runs/judge_claude-haiku-5-5.jsonl'],
          'seed': ['runs/judge_seed-oss-36b-instruct.jsonl', 'runs/judge_*-1024_seed-oss-36b-instruct.jsonl']}
QA_JUDGES = {'gemma': ['runs/judge_qa_gemma-4-31b-it-qat.jsonl'], 'seed': ['runs/judge_seed-oss-36b-instruct.jsonl']}
QA_MODELS = ['command-r', 'gpt-oss-20b', 'qwen3-30b-a3b-instruct-2507-fp8', 'mistral-small-3.2-24b-instruct']
BELEBELE_MODELS = ['claude-sonnet-5-5', 'mistral-small-3.2-24b-instruct', 'gpt-oss-20b',
                   'qwen3-30b-a3b-instruct-2507-fp8']
BELEBELE_LANGUAGES = [('bn', 'ben_Latn', 'ben_Beng'), ('ne', 'npi_Latn', 'npi_Deva')]  # language, romanized, native
CUT_OFF = {'length', 'max_tokens'}


def read_judge(patterns):
    """prompt id -> verdict, merged over every existing file matching the patterns."""
    verdicts = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            verdicts.update(read_verdicts(path))
    return verdicts


def read_finish(model):
    """prompt id -> True if the answer stopped at the token limit."""
    return {row['id']: row.get('finish') in CUT_OFF for row in read_jsonl(f'runs/outputs_{model}.jsonl')}


def belebele_records(models=BELEBELE_MODELS):
    prompts = {p['id']: p for p in read_jsonl(PROMPTS) if p['set'].startswith('belebele')}
    verdicts = {judge: read_judge(patterns) for judge, patterns in JUDGES.items()}
    english = read_jsonl(BELEBELE_ENGLISH)
    refers_to_choices = {f"{q['link']}|{q['question_number']}" for q in english
                         if re.search(r'\bfollowing\b', q['question'], re.I)}
    records = []
    for model in models:
        answers = read_answers(f'runs/outputs_{model}.jsonl')
        cut_off = read_finish(model)
        for language, romanized, native in BELEBELE_LANGUAGES:
            items = sorted(p.split('|', 2)[2] for p in prompts if p.startswith(f'belebele|{romanized}|'))
            for item in items:
                romanized_gold = prompts[f'belebele|{romanized}|{item}']['gold']
                native_gold = prompts[f'belebele|{native}|{item}']['gold']
                for question_script, config, gold, other_gold, gold_shown in [
                        ('romanized', romanized, romanized_gold, native_gold, 'both'),
                        ('native', native, native_gold, romanized_gold, 'native')]:
                    prompt_id = f'belebele|{config}|{item}'
                    judge_id = f'answer|{model}|{language}|{item}|q={question_script}|gold={gold_shown}'
                    records.append(dict(
                        set='belebele', language=language, question_script=question_script, model=model,
                        question=f'{language}|{item}', answer=answers.get(prompt_id, ''), gold=gold,
                        forms=belebele_forms(romanized_gold, native_gold), other_gold=other_gold,
                        judges={j: v[judge_id] for j, v in verdicts.items() if judge_id in v},
                        cut_off=cut_off.get(prompt_id, False), refers_to_choices=item in refers_to_choices))
    return records


def qa_records():
    """MKQA and XOR-TyDi answers of the four QA models."""
    prompts = {p['id']: p for p in read_jsonl(PROMPTS) if not p['set'].startswith('belebele')}
    with gzip.open(MKQA) as f:
        mkqa_answers = {str(record['example_id']): record['answers'] for record in map(json.loads, f)}
    verdicts = {judge: read_judge(patterns) for judge, patterns in QA_JUDGES.items()}
    records = []
    for model in QA_MODELS:
        if model == 'command-r':
            answers = {prompt_id: p.get('cmdr_response') for prompt_id, p in prompts.items()}
            cut_off = {}
        else:
            answers = read_answers(f'runs/outputs_{model}.jsonl')
            cut_off = read_finish(model)
        for prompt_id, prompt in prompts.items():
            if answers.get(prompt_id) is None:
                continue  # Command-R has no released answer for the questions without documents
            benchmark, language, example_id = prompt_id.split('|', 2)
            gold = prompt['gold']
            forms = mkqa_forms(gold, mkqa_answers[example_id], language) if benchmark == 'mkqa' else gold
            judge_id = f'qa|{model}|{prompt_id}'
            records.append(dict(
                set=benchmark, language=language, question_script='native', model=model, question=prompt_id,
                answer=answers[prompt_id] or '', gold=gold, forms=forms, other_gold=None,
                judges={j: v[judge_id] for j, v in verdicts.items() if judge_id in v},
                cut_off=cut_off.get(prompt_id, False), refers_to_choices=False))
    return records
