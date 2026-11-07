"""Clean a piece of text so that two ways of writing the same thing look the same.

Mohor never changes what an answer says. It only removes differences in how it is written:

- every digit becomes 0 to 9, so ১৭৩৬ (Bengali digits) and १७३६ (Devanagari digits) both become 1736;
- letters are put in one standard Unicode form (NFKC), since some letters can be stored in two ways;
- invisible joining characters are removed, since they change the stored text but not what a reader sees.
"""
import re
import string
import unicodedata

# Zero-width space, zero-width non-joiner, zero-width joiner, word joiner, byte-order mark.
INVISIBLE_CHARACTERS = re.compile('[​‌‍⁠﻿]')


def clean(text):
    """Return the text with ASCII digits, standard Unicode letters and no invisible characters."""
    text = ''.join(to_ascii_digit(character) for character in text)
    text = unicodedata.normalize('NFKC', text)
    return INVISIBLE_CHARACTERS.sub('', text)


def to_ascii_digit(character):
    """'৭' becomes '7'. Any character that is not a digit is returned unchanged."""
    if unicodedata.category(character) == 'Nd':
        return str(unicodedata.digit(character))
    return character


def split_words(text):
    """Lowercase the text, turn punctuation into spaces, and split it at spaces (as BERGEN does)."""
    text = text.lower()
    text = ''.join(' ' if character in string.punctuation else character for character in text)
    return text.split()
