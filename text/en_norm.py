"""Conservative English normalizer: numbers, currency, a few abbreviations.
Inline tags like [laugh] are left untouched."""
from __future__ import annotations

import re
import unicodedata

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]

ABBREV = {
    r"\bMr\.": "Mister", r"\bMrs\.": "Missus", r"\bMs\.": "Miz", r"\bDr\.": "Doctor",
    r"\bProf\.": "Professor", r"\bvs\.": "versus", r"\betc\.": "et cetera", r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is", r"\bSt\.(?= [A-Z])": "Saint",
}


def _below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    out = []
    if h:
        out.append(f"{_ONES[h]} hundred")
    if r:
        if r < 20:
            out.append(_ONES[r])
        else:
            t, o = divmod(r, 10)
            out.append(_TENS[t] + (f"-{_ONES[o]}" if o else ""))
    return " ".join(out)


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 1000:
        return _below_1000(n) if n else "zero"
    out = []
    for value, name in _SCALES:
        q, n = divmod(n, value)
        if q:
            out.append(f"{int_to_words(q)} {name}")
    if n:
        out.append(_below_1000(n))
    return " ".join(out)


def year_to_words(n: int) -> str:
    if 2000 <= n <= 2009:
        return int_to_words(n)
    if 1100 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return f"{int_to_words(hi)} hundred"
        return f"{int_to_words(hi)} {'oh ' + _ONES[lo] if lo < 10 else int_to_words(lo)}"
    return int_to_words(n)


def _ordinal(n: int) -> str:
    w = int_to_words(n)
    irregular = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                 "nine": "ninth", "twelve": "twelfth"}
    head, sep, last = w.rpartition(" ") if " " in w else ("", "", w)
    if "-" in last:
        a, b = last.rsplit("-", 1)
        last = f"{a}-{irregular.get(b, b + 'th')}"
    elif last in irregular:
        last = irregular[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}"


def _money(m: re.Match) -> str:
    sym, whole, cents = m.group(1), m.group(2).replace(",", ""), m.group(3)
    unit = {"$": "dollar", "£": "pound", "€": "euro", "৳": "taka", "₹": "rupee"}[sym]
    n = int(whole)
    s = f"{int_to_words(n)} {unit}{'' if n == 1 or unit == 'taka' else 's'}"
    if cents and int(cents):
        c = int(cents.ljust(2, "0")[:2])
        s += f" and {int_to_words(c)} cent{'s' if c != 1 else ''}" if sym in "$€" else f" {int_to_words(c)}"
    return s


def _number(m: re.Match) -> str:
    whole, frac = m.group(1).replace(",", ""), m.group(2)
    if frac is None and "," not in m.group(1) and len(whole) == 4:
        return year_to_words(int(whole))
    if len(whole) > 15:
        return " ".join(_ONES[int(c)] for c in whole)
    s = int_to_words(int(whole))
    if frac:
        s += " point " + " ".join(_ONES[int(c)] for c in frac)
    return s


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(" ", " ")
    for pat, rep in ABBREV.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r"([$£€৳₹])\s?(\d[\d,]*)(?:\.(\d{1,2}))?\b", _money, text)
    text = re.sub(r"\b(\d+)\s?%", lambda m: f"{int_to_words(int(m.group(1)))} percent", text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ordinal(int(m.group(1))), text)
    text = re.sub(r"(?<![\w.])(\d[\d,]*)(?:\.(\d+))?(?![\d])", _number, text)
    text = text.replace("&", " and ")
    return " ".join(text.split())
