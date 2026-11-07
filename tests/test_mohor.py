"""Examples from our runs, written as tests. Run: python -m pytest tests"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mohor import best_score, clean, mkqa_forms, mohor_score, published_score  # noqa: E402


def test_bengali_digits_count_as_the_same_number():
    answer = 'গ্রাফ তত্ত্ব ১৭৩৬ সালে আবিষ্কৃত হয়।'  # "Graph theory was discovered in 1736."
    assert published_score(answer, '1736') == 0.0
    assert mohor_score(answer, '1736') == 1.0


def test_clean_maps_every_digit_system_to_ascii():
    assert clean('১৭৩৬ १७३६ 1736') == '1736 1736 1736'


def test_a_name_in_english_matches_through_the_english_form():
    answer = 'ดนตรีของ "Land of Hope and Glory" ถูกแต่งโดย Edward Elgar'
    gold = ['เอ็ดเวิร์ด เอลการ์']
    mkqa_answers = {'th': [{'text': 'เอ็ดเวิร์ด เอลการ์', 'aliases': []}],
                    'en': [{'text': 'Edward Elgar', 'aliases': ['Edward William Elgar']}]}
    assert best_score(answer, gold) == 0.0
    assert best_score(answer, mkqa_forms(gold, mkqa_answers, 'th')) == 1.0


def test_romanized_answer_gets_credit_from_the_romanized_gold():
    answer = 'Hibru pariwarle adhiansh samay khula hawa bitthyo.'
    assert best_score(answer, ['खुला हावामा']) == 0.0
    assert round(best_score(answer, ['khula hawama', 'खुला हावामा']), 2) == 0.86


def test_short_word_is_found_inside_a_chinese_sentence():
    assert published_score('他的名字是老子先生', '老子') == 0.0
    assert mohor_score('他的名字是老子先生', '老子') == 1.0


def test_short_russian_word_needs_to_stand_alone():
    # "в" ("in") must not match just because a longer word contains the letter в.
    assert mohor_score('Москва большая', 'в') == 0.0
    assert mohor_score('в Москве', 'в') == 1.0


def test_short_number_matches_next_to_chinese_or_japanese_but_not_inside_a_longer_number():
    assert mohor_score('彼は10年間働いた', '10') == 1.0
    assert mohor_score('2010年に', '10') == 0.0


def test_short_devanagari_word_does_not_match_inside_a_longer_word():
    assert mohor_score('काम गर्छन्', 'छ') == 0.0
    assert mohor_score('ऊ यहाँ छ', 'छ') == 1.0


def test_a_wrong_answer_stays_wrong():
    assert mohor_score('১৭৪৫ সালে', '1736') == 0.0
