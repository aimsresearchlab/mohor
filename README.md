# mohor

Score short answers against every written form of the gold answer.

An answer key says "บ็อบ ดิลลัน". A model writes "Bob Dylan". A string score compares letters, finds none in common and
gives 0, though the answer is right. Mohor adds the other correct ways of writing the answer to the key and keeps the
same string score. It accepts the same answer written another way, never a different answer. It works for short-answer
question answering, not free-flowing text.

Mohor is early research code from the [AIMS Lab](https://aimsresearchlab.com). For now it uses the written forms a
benchmark already ships. See "Not built yet" below.

## Use

```python
from mohor import best_score, published_score

answer = 'গ্রাফ তত্ত্ব ১৭৩৬ সালে আবিষ্কৃত হয়।'      # "Graph theory was discovered in 1736."
published_score(answer, '1736')                   # 0.0: the 3-gram recall score
best_score(answer, ['1736'])                      # 1.0: Mohor

best_score('khula hawa', ['khula hawama', 'खुला हावामा'])   # best over both scripts
```

Mohor needs only Python 3.9 or later. Install it from GitHub:

```
pip install git+https://github.com/aimsresearchlab/mohor
```

The distribution is named `mohor-mrag` and the import is `mohor`. It is not on PyPI yet.

## Files

| file | what it does |
|---|---|
| `mohor/clean.py` | makes two ways of writing the same thing look the same: digits to 0 to 9, standard Unicode, no invisible characters |
| `mohor/score.py` | the published 3-gram recall score, Mohor's score, and the best score over all forms |
| `mohor/forms.py` | the written forms of a gold: so far, the forms MKQA and Belebele already ship |
| `evaluate/belebele.py` | Mohor on Belebele's romanized questions, next to the single-gold scores and the judge |
| `evaluate/mkqa.py` | each MKQA language's gap to English under the earlier fixes, Mohor and the judge |
| `evaluate/records.py` | loads every scored answer with its gold forms, judge verdicts and flags, for the scripts below |
| `evaluate/agreement.py` | answer-by-answer agreement of each scorer with the judges (AUROC, false positives and negatives) |
| `evaluate/ablation.py` | what each part of Mohor adds, with the random-question score of each step |
| `evaluate/belebele_subsets.py` | the Belebele table without the questions that need the choices, and on finished answers |
| `evaluate/invariance.py` | the script-invariance test on Belebele's answer options |
| `evaluate/short_words_and_digits.py` | short gold words and digit systems, with Mohor's rules |
| `evaluate/other_scorers.py` | chrF and LaBSE similarity next to Mohor, answer by answer against the judges |
| `evaluate/longer_answers.py` | Belebele answers made again with a 1,024-token limit, next to the first ones |
| `tests/test_mohor.py` | examples from our runs, as tests |

## Run

```
python -m pytest tests
```

The scripts in `evaluate/` are published so readers can see how each measure is computed. **They do not run from this
repository alone.** They read model answers, prompts and judge verdicts from `runs/` and `data/`, which are not released
yet. `agreement.py` and `invariance.py` also need numpy and scikit-learn.

## Checked

- `published_score` gave the same score as BERGEN's released code on all 72,364 answer-gold pairs in our runs.
- `mohor_score` was never higher than the score from our earlier short-word fix. Where it is lower, that fix had
  credited a short gold word inside a longer word: the "0" of "900.0" inside "940", or "01" inside "2018".

We have compared Mohor with LLM judges, but not yet with native-speaker labels.

## Not built yet

Transliteration into the other script, number words, a loose key for romanized spelling, Wikidata names, an LLM listing
of forms for benchmarks that ship none, and `mohor test` as a command. The test itself is in `evaluate/invariance.py`.

## License

Apache-2.0. See `LICENSE`.

`published_score` computes the 3-gram recall score used by BERGEN (Chirkova et al., 2024) so results can be compared.
This repository does not include BERGEN.
