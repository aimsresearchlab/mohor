"""Collect the written forms of a gold answer.

This version uses only the forms a benchmark already ships:

- MKQA gives, for every question, the answer in each language with its aliases, and the answer in English. The
  English answer is how most models write names ("Edward Elgar" for "เอ็ดเวิร์ด เอลการ์").
- Belebele gives each romanized question two golds: one in Latin letters and one in native script.

Planned, not built: transliteration into the other script for languages written both ways, number words
(पाँच, পাঁচ, paach for 5), Wikidata labels for names, and one LLM call per gold for sets that ship nothing.
"""

# MKQA names some languages differently from the benchmarks built on it.
MKQA_LANGUAGE_KEY = {'zh': 'zh_cn'}


def unique(forms):
    """Drop empty and repeated forms, keeping the first-seen order."""
    seen, result = set(), []
    for form in forms:
        form = (form or '').strip()
        if form and form not in seen:
            seen.add(form)
            result.append(form)
    return result


def mkqa_forms(gold, mkqa_answers, language):
    """The benchmark's gold, plus MKQA's answer and aliases in that language and in English.

    gold: the gold answers the benchmark scores against (a list of strings).
    mkqa_answers: the "answers" field of the MKQA record, a dict from language to a list of answers.
    """
    in_language = mkqa_answers.get(MKQA_LANGUAGE_KEY.get(language, language), [])
    in_english = mkqa_answers.get('en', [])
    return unique(gold + answer_texts(in_language) + answer_texts(in_english))


def answer_texts(answers):
    """Every text and alias in a list of MKQA answers."""
    texts = []
    for answer in answers:
        texts.append(answer.get('text'))
        texts.extend(answer.get('aliases') or [])
    return texts


def belebele_forms(romanized_gold, native_gold):
    """Both of Belebele's golds for a romanized question: in Latin letters and in native script."""
    return unique(romanized_gold + native_gold)
