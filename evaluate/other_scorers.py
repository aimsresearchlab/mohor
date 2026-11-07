"""Two other scorers next to Mohor, answer by answer against the judges (as agreement.py does).

- chrF (sacreBLEU's sentence chrF, default settings: character 1- to 6-grams, recall weighted twice): a
  character-string score from machine translation, used by some QA datasets. Against the benchmark's gold and
  against every written form, keeping the best.
- LaBSE: cosine similarity between multilingual sentence embeddings of the answer and the gold (Feng et al., 2022),
  which can match text across scripts. Against the benchmark's gold and against every written form.

Prints AUROC against each judge's verdict per benchmark, language and question script, with the published score
and Mohor for comparison, and a bootstrap 95% interval for LaBSE (benchmark gold) minus Mohor.

Needs sacrebleu, sentence-transformers and the LaBSE weights:
    uv run --with sacrebleu --with sentence-transformers --with scikit-learn \
        python evaluate/other_scorers.py [--wandb RUN_NAME]
"""
import sys
from collections import defaultdict

import numpy as np
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer

from agreement import BOOTSTRAP_SAMPLES, SEED, auroc
from common import log_to_wandb
from records import belebele_records, qa_records

import mohor
from mohor import best_score, published_score

CHRF_SCORER = CHRF()


def chrf(answer, gold):
    return CHRF_SCORER.sentence_score(answer, [gold]).score / 100


def embed_all(texts):
    model = SentenceTransformer('sentence-transformers/LaBSE')
    unique = sorted(set(texts))
    vectors = model.encode(unique, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    return dict(zip(unique, vectors))


def main():
    records = belebele_records() + qa_records()
    records = [r for r in records if r['judges']]
    vectors = embed_all([r['answer'] for r in records] + [f for r in records for f in r['forms'] + r['gold']])

    def labse(answer, golds):
        return max(float(vectors[answer] @ vectors[g]) for g in golds) if golds else 0.0

    for r in records:
        r['scores'] = {'published': best_score(r['answer'], r['gold'], published_score),
                       'Mohor': best_score(r['answer'], r['forms']),
                       'chrF, benchmark gold': best_score(r['answer'], r['gold'], chrf),
                       'chrF, all forms': best_score(r['answer'], r['forms'], chrf),
                       'LaBSE, benchmark gold': labse(r['answer'], r['gold']),
                       'LaBSE, all forms': labse(r['answer'], r['forms'])}

    groups = defaultdict(list)
    for r in records:
        for judge, verdict in r['judges'].items():
            groups[(judge, r['set'], r['language'], r['question_script'])].append((r, verdict))

    names = list(records[0]['scores'])
    rows = []
    print(mohor.SIGNATURE)
    print('AUROC against the judge; last column: LaBSE (benchmark gold) minus Mohor [95% bootstrap interval]')
    print(f'{"judge":6}{"set":14}{"lang":5}{"question":10}' + ''.join(f'{n[:21]:>22}' for n in names) + '   LaBSE minus Mohor')
    rng = np.random.default_rng(SEED)
    for key in sorted(groups):
        pairs = groups[key]
        labels = np.array([v for _, v in pairs])
        columns = {n: np.array([r['scores'][n] for r, _ in pairs]) for n in names}
        values = {n: auroc(labels, c) for n, c in columns.items()}
        questions = np.array([r['question'] for r, _ in pairs])
        unique_questions, index = np.unique(questions, return_inverse=True)
        draws = []
        for _ in range(BOOTSTRAP_SAMPLES):
            weights = np.bincount(rng.integers(0, len(unique_questions), len(unique_questions)),
                                  minlength=len(unique_questions))[index]
            keep = weights > 0
            draws.append(auroc(labels[keep], columns['LaBSE, benchmark gold'][keep], weights[keep])
                         - auroc(labels[keep], columns['Mohor'][keep], weights[keep]))
        low, high = np.nanpercentile(draws, [2.5, 97.5])
        difference = values['LaBSE, benchmark gold'] - values['Mohor']
        print(f'{key[0]:6}{key[1]:14}{key[2]:5}{key[3]:10}' + ''.join(f'{values[n]:>22.3f}' for n in names)
              + f'   {difference:+.3f} [{low:+.3f}, {high:+.3f}]')
        rows.append(list(key) + [len(pairs)] + [round(values[n], 4) for n in names] + [round(difference, 4),
                                                                                       round(low, 4), round(high, 4)])

    if '--wandb' in sys.argv:
        run_name = sys.argv[sys.argv.index('--wandb') + 1]
        log_to_wandb(run_name, ['judge', 'set', 'language', 'question script', 'answers'] + names +
                     ['LaBSE minus Mohor', 'low', 'high'], rows,
                     dict(signature=mohor.SIGNATURE, labse='sentence-transformers/LaBSE', chrf='sacrebleu default'))


if __name__ == '__main__':
    main()
