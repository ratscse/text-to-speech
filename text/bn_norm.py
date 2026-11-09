"""Bangla text normalizer for TTS (no external dependencies).

* Unicode: NFC, strip zero-width spaces/BOM (ZWJ/ZWNJ kept — they shape
  conjuncts), unify nukta letters (ড় ঢ় য়) to their precomposed forms, fix
  split vowel signs (ে + া -> ো, ে + ৗ -> ৌ), drop a duplicated hasant.
* Numbers (Bangla ০-৯ or ASCII digits) -> Bangla words with the Indian
  grouping (হাজার, লাখ, কোটি); 4-digit years 1100-1999 read as "উনিশশো একাত্তর";
  decimals "দশমিক"; long digit strings (phones, IDs) digit by digit.
* Symbols: % ৳ $ & + = etc. -> words. Punctuation: '|' -> '।', tidy spacing.
"""
from __future__ import annotations

import re
import unicodedata

ONES = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
    "বিশ", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "উনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "উনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "উনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "উনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "উনসত্তর",
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "উনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "আটাশি", "উননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]
assert len(ONES) == 100

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_TO_ASCII = str.maketrans(BN_DIGITS, "0123456789")

SYMBOLS = {
    "%": " শতাংশ", "৳": " টাকা", "$": " ডলার", "€": " ইউরো", "£": " পাউন্ড", "₹": " রুপি",
    "&": " এবং ",
}
_CURRENCY_PREFIX_RE = re.compile(r"([$৳€£₹])\s?(\d[\d,]*(?:\.\d+)?)")


def _hundreds(n: int) -> str:
    """0 < n < 1000"""
    h, r = divmod(n, 100)
    parts = []
    if h:
        parts.append(ONES[h] + "শো" if h != 1 else "একশো")
    if r:
        parts.append(ONES[r])
    return " ".join(parts)


def int_to_words(n: int) -> str:
    if n < 0:
        return "মাইনাস " + int_to_words(-n)
    if n < 100:
        return ONES[n]
    parts = []
    crore, n = divmod(n, 10_000_000)
    if crore:
        parts.append(int_to_words(crore) + " কোটি")
    lakh, n = divmod(n, 100_000)
    if lakh:
        parts.append(ONES[lakh] + " লাখ")
    hajar, n = divmod(n, 1000)
    if hajar:
        parts.append(ONES[hajar] + " হাজার")
    if n:
        parts.append(_hundreds(n))
    return " ".join(parts)


def year_to_words(n: int) -> str:
    """1971 -> উনিশশো একাত্তর; outside 1100-1999 read as a normal number."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return ONES[hi] + "শো" + (" " + ONES[lo] if lo else "")
    return int_to_words(n)


def digits_to_words(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s)


_NUM_RE = re.compile(r"(?<![\w.])([-−]?)(\d[\d,]*)(?:\.(\d+))?(?![\d])")


def _num_sub(m: re.Match) -> str:
    sign, whole, frac = m.group(1), m.group(2), m.group(3)
    digits = whole.replace(",", "")
    if not digits:
        return m.group(0)
    if len(digits) > 9 and "," not in whole:  # phone numbers, IDs
        words = digits_to_words(digits)
    elif len(digits) == 4 and "," not in whole and frac is None and not sign:
        words = year_to_words(int(digits))
    elif digits.startswith("0") and len(digits) > 1:
        words = digits_to_words(digits)
    else:
        words = int_to_words(int(digits))
    if frac:
        words += " দশমিক " + digits_to_words(frac)
    if sign:
        words = "মাইনাস " + words
    return words  # no padding: "১০টা" -> "দশটা"


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "").replace(" ", " ")
    # precomposed nukta letters (NFC decomposes these; models expect one form)
    text = text.replace("ড়", "ড়").replace("ঢ়", "ঢ়").replace("য়", "য়")
    # split vowel signs
    text = text.replace("ো", "ো").replace("ৌ", "ৌ")
    # doubled hasant / doubled vowel signs from bad typing
    text = re.sub("্{2,}", "্", text)
    text = re.sub(r"([া-ৌ])\1+", r"\1", text)
    return text


def normalize_punct(text: str) -> str:
    text = text.replace("|", "।").replace("॥", "।")
    text = re.sub(r"[“”„«»]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\s*([।,;:?!])", r"\1", text)  # no space before punctuation
    text = re.sub(r"([।?!])(?=\S)", r"\1 ", text)  # space after sentence end
    text = re.sub(r"([।?!.]){2,}", r"\1", text)
    return " ".join(text.split())


def normalize(text: str) -> str:
    text = normalize_unicode(text)
    text = text.translate(_TO_ASCII)
    text = _CURRENCY_PREFIX_RE.sub(lambda m: m.group(2) + m.group(1), text)  # "৳500" -> "500৳" -> words + টাকা
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, word)
    text = _NUM_RE.sub(_num_sub, text)
    return normalize_punct(text)
