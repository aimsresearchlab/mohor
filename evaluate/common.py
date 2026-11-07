"""Small helpers shared by the evaluation scripts. Run the scripts from the repository root."""
import json
import os
import random
import re
import subprocess
import sys

# Let the scripts import the mohor package from the folder next to this one.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

VERDICT = re.compile(r'\b(incorrect|correct)\b')
RANDOM_SEED = 0


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_answers(path):
    """prompt id -> the model's answer, from a runs/outputs_<model>.jsonl file."""
    return {row['id']: row.get('response') or '' for row in read_jsonl(path)}


def read_verdicts(path):
    """judge prompt id -> True if the judge said "correct". Prompts without a clear verdict are left out."""
    verdicts = {}
    for row in read_jsonl(path):
        match = VERDICT.search((row.get('response') or '').lower())
        if match:
            verdicts[row['id']] = match.group(1) == 'correct'
    return verdicts


def another_question(questions):
    """Pair every question with another one, the same way every time (for the random-question check)."""
    shuffled = questions[:]
    random.Random(RANDOM_SEED).shuffle(shuffled)
    return dict(zip(questions, shuffled))


def percent(total, count):
    return 100 * total / count if count else 0.0


def log_to_wandb(run_name, columns, rows, settings):
    """Log one results table to W&B (project mohor), online. The key comes from the environment."""
    import wandb
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    run = wandb.init(project='mohor', group='evaluate', name=run_name, id=run_name, resume='allow',
                     config=dict(settings, git_commit=commit))
    run.log({'results': wandb.Table(columns=columns, data=rows)})
    run.finish()
