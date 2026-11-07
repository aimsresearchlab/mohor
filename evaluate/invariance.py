"""The script-invariance test: does a scorer's accuracy change with the script of the answer or of the gold?

It uses Belebele's own answer options for the romanized questions, so whether an answer is right is known: the
correct option and one wrong option, each written by Belebele's annotators in native script and in Latin
letters. Every pairing of answer script and gold script is scored. A scorer that ignores script gets the same
accuracy in every row.

Scorers (an answer counts as correct at a score of 0.5 or more):
- published: BERGEN's 3-gram recall against the gold shown;
- earlier fixes: Mohor's cleaning and short-word rule against the gold shown only;
- Mohor: against both of Belebele's golds, whichever is shown (Mohor always uses every form it has);
- the Gemma 4 31B and Claude Haiku 5.5 judges, given the same prompt (from runs/judge_*.jsonl).

Usage (from the repository root): python evaluate/invariance.py
"""
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from common import percent, read_jsonl
from records import JUDGES, read_judge

import mohor
from mohor import best_score, published_score

JUDGE_INPUTS = 'runs/judge_inputs_2026-10-27.jsonl'
THRESHOLD = 0.5
SCORERS = ['published', 'earlier fixes', 'Mohor'] + list(JUDGES)


def golds_shown(user_text):
    """The gold answers written in a judge prompt (one, or two for 'both')."""
    block = user_text.split('Correct answer', 1)[1].split("Model's answer:", 1)[0]
    if block.startswith(':'):
        return [block[1:].strip()]
    return [line[2:].strip() for line in block.splitlines() if line.startswith('- ')]


def main():
    prompts = [p for p in read_jsonl(JUDGE_INPUTS) if p['part'] == 'check' and p['question_script'] == 'romanized']
    verdicts = {judge: read_judge(patterns) for judge, patterns in JUDGES.items()}
    both_golds = {(p['lang'], p['item']): golds_shown(p['user']) for p in prompts if p['gold_shown'] == 'both'}

    cells = defaultdict(lambda: defaultdict(list))  # (language, answer script, gold shown) -> scorer -> (label, value)
    for p in prompts:
        answer = re.search(r"Model's answer: (.*)\Z", p['user'], re.S).group(1).strip()
        gold = golds_shown(p['user'])
        label = p['label'] == 'correct'
        key = (p['lang'], p['answer_script'], p['gold_shown'])
        values = {'published': best_score(answer, gold, published_score),
                  'earlier fixes': best_score(answer, gold),
                  'Mohor': best_score(answer, both_golds[(p['lang'], p['item'])])}
        for judge in JUDGES:
            if p['id'] in verdicts[judge]:
                values[judge] = float(verdicts[judge][p['id']])
        for scorer, value in values.items():
            cells[key][scorer].append((label, value))

    print(mohor.SIGNATURE)
    print(f'Accuracy in percent (score >= {THRESHOLD} counts as correct); AUROC in brackets for the string scores.')
    for language in ['bn', 'ne']:
        print(f'\n{language}: romanized questions, 900 right and 900 wrong options per row')
        print(f'{"answer script":>14}{"gold shown":>12}' + ''.join(f'{s:>22}' for s in SCORERS))
        spread = defaultdict(list)
        for answer_script in ['romanized', 'native']:
            for gold_shown in ['romanized', 'native', 'both']:
                row = cells[(language, answer_script, gold_shown)]
                text = f'{answer_script:>14}{gold_shown:>12}'
                for scorer in SCORERS:
                    pairs = row.get(scorer)
                    if not pairs:
                        text += f'{"-":>22}'
                        continue
                    labels = np.array([label for label, _ in pairs])
                    values = np.array([value for _, value in pairs])
                    accuracy = percent(np.sum((values >= THRESHOLD) == labels), len(pairs))
                    spread[scorer].append(accuracy)
                    if scorer in JUDGES:
                        text += f'{accuracy:>22.1f}'
                    else:
                        text += f'{accuracy:>13.1f} [{roc_auc_score(labels, values):.3f}]'
                print(text)
        print(f'{"spread (max - min)":>26}' + ''.join(f'{max(spread[s]) - min(spread[s]):>22.1f}' if spread[s] else
                                                     f'{"-":>22}' for s in SCORERS))


if __name__ == '__main__':
    main()
