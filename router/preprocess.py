"""Classical NLP front end (technique area B).

* regex text normalization (unicode, case, repeated letters, URLs, e-mails, numbers)
* regex extraction of product model codes such as "DW-6534" or "WM 1200X"
* minimum-edit-distance (Levenshtein, dynamic programming) spelling correction of
  out-of-vocabulary words against a lexicon built from the training data plus a
  product-name list
* ELIZA-style pattern/response rules for greetings and small talk
"""
from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter

# --------------------------------------------------------------------------- #
# Regex normalization and model-code extraction
# --------------------------------------------------------------------------- #
# "DW-6534", "DW6534", "WM 1200X" (upper-case prefix) or "dw-6534" / "dw6534"
MODEL_CODE_RE = re.compile(
    r"\b(?:([A-Z]{2,4})[- ]?(\d{3,6}[A-Z]{0,2})|([a-z]{1,4})-?(\d{3,6}[a-z]{0,2}))\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
REPEAT_RE = re.compile(r"(\w)\1{2,}")          # "sooooo" -> "soo"
NON_TEXT_RE = re.compile(r"[^a-z0-9' ]+")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9']+")


def extract_model_codes(text: str) -> list[str]:
    """Return product model codes in canonical form, e.g. 'dw 6534' -> 'DW-6534'."""
    codes = []
    for up, upnum, low, lownum in MODEL_CODE_RE.findall(text):
        prefix, num = (up, upnum) if up else (low, lownum)
        codes.append(f"{prefix.upper()}-{num.upper()}")
    return codes


def normalize(text: str) -> str:
    """Deterministic normalization used for *every* model (train and test)."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").lower()
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = MODEL_CODE_RE.sub(" ", t)          # model codes are extracted separately
    t = REPEAT_RE.sub(r"\1\1", t)
    t = NON_TEXT_RE.sub(" ", t)
    return SPACE_RE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


# --------------------------------------------------------------------------- #
# Minimum edit distance spelling correction
# --------------------------------------------------------------------------- #
def edit_distance(a: str, b: str, max_d: int | None = None) -> int:
    """Levenshtein distance (insert/delete/substitute cost 1) via DP.

    If max_d is given, stops early and returns max_d + 1 once every cell of a row
    exceeds max_d.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if max_d is not None and min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


PRODUCT_NAMES = [
    "dishwasher", "refrigerator", "fridge", "freezer", "oven", "microwave",
    "washer", "dryer", "thermostat", "vacuum", "kettle", "toaster", "blender",
    "television", "speaker", "router", "printer", "lights", "alarm",
]


def english_words(n: int = 60000) -> set[str]:
    """The n most frequent English words (wordfreq package)."""
    from wordfreq import top_n_list

    return set(top_n_list("en", n))


class SpellCorrector:
    """Corrects unknown tokens to the closest lexicon word.

    A token is left unchanged if it is in the training vocabulary or in a general English
    word list (so valid words such as "drain" are never "corrected"). Otherwise the
    candidates are lexicon words (training words seen >= min_count times + product names)
    within edit distance 1 (tokens of length 4-7) or 2 (length >= 8); ties are broken by
    corpus frequency.
    """

    def __init__(self, corpus: list[str], extra_words: list[str] = PRODUCT_NAMES, min_count: int = 2,
                 known_words: set[str] | None = None):
        counts = Counter(w for s in corpus for w in tokenize(s))
        for w in extra_words:
            counts[w] += min_count
        self.freq = {w: c for w, c in counts.items() if c >= min_count}
        self.known = set(counts) | set(known_words or ())
        self.by_len: dict[int, list[str]] = {}
        for w in self.freq:
            self.by_len.setdefault(len(w), []).append(w)
        self._cache: dict[str, str] = {}

    def correct_word(self, w: str) -> str:
        if w in self.known or len(w) < 4 or not w.isalpha():
            return w
        if w in self._cache:
            return self._cache[w]
        max_d = 1 if len(w) < 8 else 2
        best, best_key = w, (max_d + 1, 0)
        for L in range(len(w) - max_d, len(w) + max_d + 1):
            for cand in self.by_len.get(L, []):
                d = edit_distance(w, cand, max_d)
                if d <= max_d:
                    key = (d, -self.freq[cand])
                    if key < best_key:
                        best, best_key = cand, key
        self._cache[w] = best
        return best

    def correct(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Return corrected text and the list of (wrong, corrected) pairs."""
        out, changes = [], []
        for w in text.split():
            c = self.correct_word(w)
            if c != w:
                changes.append((w, c))
            out.append(c)
        return " ".join(out), changes


# --------------------------------------------------------------------------- #
# ELIZA-style small talk
# --------------------------------------------------------------------------- #
ELIZA_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"^(hi+|hello+|hey+|good (morning|afternoon|evening)|greetings)\b"),
     ["Hello! How can I help you with your request today?",
      "Hi there! Tell me what you need help with."]),
    (re.compile(r"\bhow are you\b|\bhow's it going\b|\bhow are things\b"),
     ["I'm doing well, thanks for asking! What can I do for you?"]),
    (re.compile(r"\b(thank you|thanks|thx|cheers)\b"),
     ["You're welcome! Is there anything else I can help with?"]),
    (re.compile(r"^(bye|goodbye|see you|see ya|good night)\b"),
     ["Goodbye! Have a great day."]),
    (re.compile(r"\bi (feel|am feeling|am) (sad|bored|tired|lonely|happy|great)\b"),
     ["Why do you feel {2}?", "How long have you been feeling {2}?"]),
    (re.compile(r"\b(tell me a joke|say something funny|make me laugh)\b"),
     ["Why did the dishwasher break up with the fridge? It needed some space to cool down."]),
    (re.compile(r"\bare you (a )?(bot|robot|human|real|ai)\b|\bwho are you\b|\bwhat is your name\b"),
     ["I'm the automated support assistant. I route your message to the right team."]),
]
SMALL_TALK_FALLBACK = ["I see. Could you tell me a bit more about how I can help?",
                       "Interesting! Is there something I can help you with today?"]


def eliza_reply(text: str, rng: random.Random | None = None, max_extra: int = 3) -> str | None:
    """Return a rule-based reply if the message is *only* small talk, else None.

    A pattern counts only if at most `max_extra` other words remain after removing it, so
    "hello, how are you?" gets a reply but "hi, my card got declined" is routed normally.
    """
    rng = rng or random.Random(0)
    t = normalize(text)
    for pattern, responses in ELIZA_RULES:
        m = pattern.search(t)
        if m and len(tokenize(pattern.sub(" ", t))) <= max_extra:
            groups = [m.group(0)] + list(m.groups())
            resp = rng.choice(responses)
            return resp.format(*[g or "" for g in groups])
    return None


def small_talk_reply(text: str) -> str:
    return eliza_reply(text) or SMALL_TALK_FALLBACK[0]


class Preprocessor:
    """Full front end: normalize -> extract model codes -> spelling correction."""

    def __init__(self, corrector: SpellCorrector | None):
        self.corrector = corrector

    def __call__(self, text: str) -> dict:
        codes = extract_model_codes(text)
        norm = normalize(text)
        changes: list = []
        if self.corrector is not None:
            norm, changes = self.corrector.correct(norm)
        return {"text": norm, "model_codes": codes, "corrections": changes}
