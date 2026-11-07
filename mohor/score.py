"""Score an answer against a gold answer, and against every written form of it.

The base score is the one multilingual RAG papers report: BERGEN's character 3-gram recall (Chirkova et al.,
2024). It cuts each gold word into overlapping 3-letter pieces ("tokyo" gives "tok", "oky", "kyo") and counts the
share of those pieces that also appear in the answer. Gold words shorter than 3 letters are kept whole.

`published_score` is that score, unchanged. `mohor_score` is the same score after two fixes:

1. Both texts are cleaned first (see clean.py): digits, Unicode form, invisible characters.
2. A short gold word (under 3 letters) is looked for properly. BERGEN only finds it if the answer has the same
   word standing alone, so 老子 is never found in a Chinese sentence, which has no spaces. Mohor looks for it inside
   the answer's words in scripts that write words together (Chinese, Japanese, Korean, Thai and their
   neighbours). Everywhere else it must stand apart from the letters around it, so the Russian word "в" ("in")
   does not match any text that happens to contain the letter в, and "10" does not match "2010".

`best_score` gives an answer its best score over every form of the gold. That is the whole idea of Mohor.
"""
import unicodedata
from collections import Counter

from .clean import clean, split_words

PIECE_LENGTH = 3

# Scripts that write words together, without spaces between them (or, for Korean, with grammar attached to the
# word). In these, a short gold word has to be found inside the answer's words. The names are the start of the
# Unicode character names.
SCRIPTS_WITHOUT_WORD_SPACES = ('CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL', 'THAI', 'LAO', 'KHMER', 'MYANMAR')


def pieces(text):
    """The 3-letter pieces of every word; words shorter than 3 letters are kept whole (as BERGEN does)."""
    result = []
    for word in split_words(text):
        if len(word) < PIECE_LENGTH:
            result.append(word)
        else:
            result.extend(word[i:i + PIECE_LENGTH] for i in range(len(word) - PIECE_LENGTH + 1))
    return result


def published_score(answer, gold):
    """BERGEN's 3-gram recall: the share of the gold's pieces that also appear in the answer."""
    gold_pieces = pieces(gold)
    if not gold_pieces:
        return 0.0
    shared = Counter(pieces(answer)) & Counter(gold_pieces)
    return sum(shared.values()) / len(gold_pieces)


def mohor_score(answer, gold):
    """3-gram recall after cleaning both texts, with short gold words looked for properly."""
    answer, gold = clean(answer), clean(gold)
    answer_words = split_words(answer)
    unused_answer_pieces = Counter(pieces(answer))

    needed = found = 0
    for word in split_words(gold):
        if len(word) < PIECE_LENGTH:
            needed += 1
            found += short_word_is_in(word, answer_words)
            continue
        for i in range(len(word) - PIECE_LENGTH + 1):
            piece = word[i:i + PIECE_LENGTH]
            needed += 1
            if unused_answer_pieces[piece] > 0:
                found += 1
                unused_answer_pieces[piece] -= 1  # each answer piece can match only once
    return found / needed if needed else 0.0


def short_word_is_in(word, answer_words):
    """Is a short gold word in the answer?

    A word from a script without word spaces (老子) counts anywhere inside the answer's words. Any other short
    word (10, в, US) counts only where no letter or digit of a spaced script touches it: "10" is found in "10年"
    and in "10", but not in "2010"; "в" is found in "в Москве" but not inside "Москва".
    """
    if written_without_spaces(word):
        return any(word in answer_word for answer_word in answer_words)
    return any(stands_apart(word, answer_word) for answer_word in answer_words)


def stands_apart(word, answer_word):
    start = answer_word.find(word)
    while start != -1:
        end = start + len(word)
        before = answer_word[start - 1] if start > 0 else ''
        after = answer_word[end] if end < len(answer_word) else ''
        if not continues_the_word(before) and not continues_the_word(after):
            return True
        start = answer_word.find(word, start + 1)
    return False


def continues_the_word(character):
    """A letter, digit or vowel sign of a script that separates words with spaces."""
    if not character or written_without_spaces(character):
        return False
    return unicodedata.category(character)[0] in 'LNM'  # letter, number, or mark (such as a Devanagari vowel sign)


def written_without_spaces(text):
    return any(unicodedata.name(character, '').startswith(SCRIPTS_WITHOUT_WORD_SPACES) for character in text)


def best_score(answer, gold_forms, score=mohor_score):
    """The answer's best score over every written form of the gold."""
    return max((score(answer, form) for form in gold_forms), default=0.0)
