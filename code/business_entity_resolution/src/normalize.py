"""Language-agnostic normalization (no country lists).

normalize_text(s):
  1. Unicode NFKC.
  2. Every Indic-script run (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
     Malayalam) is transliterated to Latin with indic-transliteration (scheme -> IAST), word by word
     (cached). Word-final bare consonants get an explicit virama first (drops the written-but-unspoken
     final schwa, e.g. "प्रोजेक्ट्स" -> "projekts" instead of "projektsa"); anusvara/candrabindu become
     "n" ("m" before p/b/m). Any failure falls back to unidecode.
  3. unidecode for everything else (accents, other scripts), lowercase.
  4. Web/domain cleanup (www., .com/.in/...), "&" -> "and", punctuation -> space, leading zeros of
     numbers stripped, leet digits inside long alphabetic tokens (c0nstructions -> constructions).
  5. Generic abbreviation expansion (a small cross-country list of legal-form and address
     abbreviations, see ABBREV). Frequency-based stopwords are learned from the data separately
     (``learn_stopwords``).
Also: ``script_of`` (dominant script of a string, "Latin" if no Indic letters), ``number_tokens``
(digit runs of an address), ``skeleton`` (consonant skeleton used as a cross-script phonetic key).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

from unidecode import unidecode

try:
    from indic_transliteration import sanscript
    _HAVE_SANSCRIPT = True
except Exception:  # pragma: no cover
    _HAVE_SANSCRIPT = False

# Unicode block base -> (script name, sanscript scheme name). All nine blocks share the ISCII layout.
INDIC_BLOCKS = {
    0x0900: ("Devanagari", "DEVANAGARI"), 0x0980: ("Bengali", "BENGALI"), 0x0A00: ("Gurmukhi", "GURMUKHI"),
    0x0A80: ("Gujarati", "GUJARATI"), 0x0B00: ("Oriya", "ORIYA"), 0x0B80: ("Tamil", "TAMIL"),
    0x0C00: ("Telugu", "TELUGU"), 0x0C80: ("Kannada", "KANNADA"), 0x0D00: ("Malayalam", "MALAYALAM"),
}
SCRIPTS = ["Latin"] + [v[0] for v in INDIC_BLOCKS.values()]
_INDIC_RUN = re.compile(r"[\u0900-\u0D7F]+(?:[\s\u200c\u200d]+[\u0900-\u0D7F]+)*")


def _block(ch: str):
    o = ord(ch)
    if 0x0900 <= o <= 0x0D7F:
        return o & ~0x7F
    return None


def script_of(s: str) -> str:
    """Dominant Indic script by letter count, 'Latin' if the string has no Indic characters."""
    c = Counter(b for b in map(_block, s) if b is not None)
    if not c:
        return "Latin"
    return INDIC_BLOCKS[c.most_common(1)[0][0]][0]


def _is_consonant(o: int, base: int) -> bool:
    off = o - base
    return 0x15 <= off <= 0x39 or 0x58 <= off <= 0x5F


def _fix_candra(word: str) -> str:
    """Candra o/e (offsets 0x49/0x45, used for English loanwords) -> plain o/e signs."""
    out = []
    for ch in word:
        o = ord(ch)
        b = o & ~0x7F
        if b in INDIC_BLOCKS and (o - b) in (0x49, 0x45):
            ch = chr(o + 2)
        elif b in INDIC_BLOCKS and (o - b) in (0x11, 0x0D):  # independent candra o / candra e
            ch = chr(o + 2)
        out.append(ch)
    return "".join(out)


_ML_CHILLU = {0x0D7A: "\u0D23", 0x0D7B: "\u0D28", 0x0D7C: "\u0D30", 0x0D7D: "\u0D32", 0x0D7E: "\u0D33",
              0x0D7F: "\u0D15", 0x0D54: "\u0D2E", 0x0D55: "\u0D2F", 0x0D56: "\u0D34"}


def _prep(word: str, base: int) -> tuple[str, int]:
    """Script-specific fixes before IAST transliteration. Returns (word, base to transliterate with).

    * Malayalam chillu letters -> consonant + virama.
    * Tamil is shifted onto the parallel Devanagari code points (same ISCII offsets): Tamil writes
      only unvoiced stops, and the sanscript TAMIL scheme renders them as aspirated/voiced letters."""
    if base == 0x0D00:
        word = "".join(_ML_CHILLU[ord(c)] + "\u0D4D" if ord(c) in _ML_CHILLU else c for c in word)
    if base == 0x0B80:
        word = "".join(chr(ord(c) - 0x0280) if 0x0B80 <= ord(c) <= 0x0BFF else c for c in word)
        base = 0x0900
    return word, base


def _add_final_virama(word: str) -> str:
    word = _fix_candra(word)
    if not word:
        return word
    o = ord(word[-1])
    base = o & ~0x7F
    if base in INDIC_BLOCKS and _is_consonant(o, base) and len(word) > 1:
        return word + chr(base + 0x4D)
    return word


_ANUSVARA_LAB = re.compile(r"[ṃṁ](?=[pbm])")
_ANUSVARA = re.compile(r"[ṃṁ]")


@lru_cache(maxsize=500_000)
def translit_word(word: str) -> str:
    """Transliterate one Indic word (single script) to ASCII Latin."""
    base = None
    for ch in word:
        b = _block(ch)
        if b is not None:
            base = b
            break
    if base is None:
        return unidecode(word)
    try:
        if not _HAVE_SANSCRIPT:
            raise RuntimeError
        orig_base = base
        word, base = _prep(word, base)
        scheme = getattr(sanscript, INDIC_BLOCKS[base][1])
        out = sanscript.transliterate(_add_final_virama(word), scheme, sanscript.IAST)
        if orig_base == 0x0B80:  # Tamil: ca is mostly /s/ in loanwords
            out = out.replace("c", "s")
        if orig_base == 0x0D00:  # Malayalam: double alveolar r is /tt/
            out = out.replace("ṟṟ", "tt")
        out = _ANUSVARA.sub("n", _ANUSVARA_LAB.sub("m", out))
        out = out.replace("ś", "sh").replace("ṣ", "sh").replace("Ś", "sh").replace("Ṣ", "sh")
        return unidecode(out)
    except Exception:
        return unidecode(word)


def transliterate_indic(s: str) -> str:
    if not _INDIC_RUN.search(s):
        return s
    return _INDIC_RUN.sub(lambda m: " ".join(translit_word(w) for w in m.group(0).split()), s)


ABBREV = {
    # legal forms / generic business words (cross-country)
    "pvt": "private", "prv": "private", "ltd": "limited", "ltda": "limited", "lmt": "limited",
    "corp": "corporation", "inc": "incorporated", "incorp": "incorporated", "co": "company",
    "cos": "companies", "intl": "international", "natl": "national", "mfg": "manufacturing",
    "svc": "services", "svcs": "services", "srvcs": "services", "bros": "brothers",
    "assoc": "associates", "assocs": "associates", "ent": "enterprises", "mgmt": "management",
    "tech": "technologies", "techs": "technologies", "grp": "group", "dept": "department",
    "univ": "university", "hosp": "hospital", "ctr": "center", "centre": "center", "and": "and",
    # address
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
    "ln": "lane", "ct": "court", "blvd": "boulevard", "hwy": "highway", "pkwy": "parkway",
    "pl": "place", "sq": "square", "cir": "circle", "trl": "trail", "ter": "terrace", "mt": "mount",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "nr": "near", "opp": "opposite", "sec": "sector", "ph": "phase", "hno": "house", "hn": "house",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "no": "number",
}
_WEB = re.compile(r"\bwww\.|\.(?:com|co\.in|in|net|org|biz|info|fr|us|io)\b")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_LEADING_ZERO = re.compile(r"\b0+(\d)")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def _deleet(tok: str) -> str:
    nd = sum(ch.isdigit() for ch in tok)
    if nd and nd <= 2 and len(tok) - nd >= 4 and tok[0].isalpha():
        return tok.translate(_LEET)
    return tok


def normalize_text(s: str, expand: bool = True) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = transliterate_indic(s)
    s = unidecode(s).lower()
    s = _WEB.sub(" ", s).replace("&", " and ").replace("h.no", " hno ")
    s = _NONALNUM.sub(" ", s)
    s = _LEADING_ZERO.sub(r"\1", s)
    toks = [_deleet(t) for t in s.split()]
    if expand:
        toks = [ABBREV.get(t, t) for t in toks]
    return " ".join(toks)


_DIGITS = re.compile(r"\d+")


def number_tokens(norm: str) -> list[str]:
    return _DIGITS.findall(norm)


_SK_MAP = str.maketrans({"c": "k", "q": "k", "z": "s", "v": "b", "w": "b", "x": "k", "j": "g"})
_SOFT_C = re.compile(r"c(?=[eiy])")
_VOWELS = re.compile(r"[aeiouyh]")
_DUP = re.compile(r"(.)\1+")


def skeleton(norm: str) -> str:
    """Consonant skeleton per token (vowels/h/y removed, ph->f, soft c->s, c/q/x->k, z->s, v/w->b, j->g, repeats collapsed)."""
    out = []
    for t in norm.split():
        if t.isdigit():
            out.append(t)
            continue
        t = _SOFT_C.sub("s", t.replace("ph", "f"))
        k = _DUP.sub(r"\1", _VOWELS.sub("", t.translate(_SK_MAP)))
        out.append(k or t[:1])
    return " ".join(out)


def learn_stopwords(texts, top_n: int = 50) -> list[str]:
    """Top-N most frequent tokens by document frequency (data-driven legal-suffix/stopword list)."""
    df = Counter()
    for t in texts:
        df.update(set(t.split()))
    return [w for w, _ in df.most_common(top_n)]
