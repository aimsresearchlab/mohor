"""Mohor: score short answers against every written form of the gold answer.

    from mohor import best_score
    best_score('গ্রাফ তত্ত্ব ১৭৩৬ সালে আবিষ্কৃত হয়।', ['1736'])   # 1.0; the published score gives 0.0
"""
from .clean import clean
from .forms import belebele_forms, mkqa_forms
from .score import best_score, mohor_score, published_score

__version__ = '0.0.1'

# Printed next to every reported number, so a reader knows exactly what was run.
SIGNATURE = f'mohor-{__version__}|base:3gram-recall|clean:digits+nfkc+joiners|short-words:by-script'
