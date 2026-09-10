#!/usr/bin/env python3
"""Orderbook attrition calculator.

Implements prompt_v2_impl.md via prompt_v2_pseudocode.md. Section references (§)
point at prompt_v2_impl.md.

Usage:
    python attrition_calculator.py INPUT.json [-o OUTPUT.json] [--test] [--summary]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

NA = "N/A"
Value = Union[float, str]  # a number, or the string "N/A"


# --------------------------------------------------------------------------
# §1  Constants and patterns
# --------------------------------------------------------------------------

CURRENCY_ALIASES = {
    "€": "EUR", "EUR": "EUR",
    "£": "GBP", "GBP": "GBP",
    "US$": "USD", "USD": "USD", "$": "USD",
    "A$": "AUD", "AUD": "AUD",
    "S$": "SGD", "SGD": "SGD",
    "HK$": "HKD", "HKD": "HKD",
    "NZ$": "NZD", "NZD": "NZD",
    "C$": "CAD", "CAD": "CAD",
    "¥": "JPY", "JPY": "JPY",
    "RMB": "CNY", "CNY": "CNY",
    "CNH": "CNH", "CHF": "CHF", "SEK": "SEK", "NOK": "NOK", "DKK": "DKK",
}
# longest-first so "US$" and "A$" beat "$", and "CNH" beats nothing shorter
_CUR_KEYS = sorted(CURRENCY_ALIASES, key=len, reverse=True)
CUR_RE = "|".join(re.escape(k) for k in _CUR_KEYS)

# Units, including the misspellings people actually produce. Observed forms are
# marked; the rest are the same slips one keystroke away and are listed so the next
# extract does not need a code change. Nothing here may collide with a word or bank
# name that can follow a number — see the guard test in `check_unit_collisions`.
BILLION_UNITS = [
    # observed
    "billions", "billion", "blns", "bln", "bns", "bnm", "bnn", "bnv", "bio",
    "bill", "bil", "bm", "bn", "b", "n",
    # plausible human variants
    "billons", "billon", "bilions", "bilion", "bilon", "bios", "blm", "bnz",
    "bnss", "bnd",
]
MILLION_UNITS = [
    # observed
    "millions", "million", "mlns", "mln", "mns", "mio", "mill", "mil", "mmn",
    "mmm", "mm", "mn", "ml", "m",
    # plausible human variants
    "milions", "milion", "millons", "millon", "mios", "mils", "mms", "mmns",
    "mlm", "mnn", "mnm",
]
THOUSAND_UNITS = ["thousands", "thousand", "ths", "th", "k"]

UNIT_MULTIPLIER: dict[str, float] = {}
for _u in BILLION_UNITS:
    UNIT_MULTIPLIER[_u] = 1000.0
for _u in MILLION_UNITS:
    UNIT_MULTIPLIER[_u] = 1.0
for _u in THOUSAND_UNITS:
    UNIT_MULTIPLIER[_u] = 0.001
_UNIT_KEYS = sorted(UNIT_MULTIPLIER, key=len, reverse=True)

TIMESTAMP = re.compile(r"<\d{10,}>")
JUNK_TOKEN = re.compile(r"^\d{10,}$")

DEAL_MARKERS = re.compile(
    r"\b(?:combined|total|aggregate)\b"
    r"|across\s+(?:the\s+)?(?:both\s+|all\s+|\d+\s+)?tranches",
    re.I,
)

MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
BOND_LABEL = re.compile(
    r"(?:interest\s+split\s*:)"
    r"|\b(?:FXD-to-FRN|FXD-FRN|FXD|FRN|FtF|Fixed|Tap)\b"
    r"|\bNew\s+\d+(?:\.\d+)?\s*(?:y|yr|yrs|year|years)\b"
    r"|\b\d+(?:\.\d+)?\s*-?\s*(?:y|yr|yrs|year|years)\b"
    r"|\b\d+NC\d+\b"
    r"|\b\d{2}s\b"
    r"|\b(?:" + MONTHS + r")[-\s]?\d{2,4}\b",
    re.I,
)

# no word boundaries: must also match "...2.1bnFinal book..."
CLOSING = re.compile(
    r"final"
    r"|books?\s*clos"
    r"|clos(?:ed)?\s+(?:north of|in excess of|at)"
    r"|book\s+close",
    re.I,
)

BOILERPLATE = re.compile(
    r"joint lead manager"
    r"|bookrunners?\s*:"
    r"|placing agent"
    r"|bookbuilding and placing"
    r"|stabilisation"
    r"|code of conduct"
    r"|mifid"
    r"|target market"
    r"|not for distribution",
    re.I,
)
BOILERPLATE_MAX_LEN = 200

LEAD_KEYWORDS = (
    r"JLM|JBR|LM\s+interest|lead|bookrunner|prop|SFC"
)

# §5.3 — every pattern here is masked (blanked, length preserved) before parsing.
NOISE_PATTERNS = [
    re.compile(r"<\d{10,}>"),                                        # timestamps
    # Any number followed by Y/YR/YRS/YEAR is a tenor: 5yr, 10-year, 11Y:, 6Y,
    # and decimals — 6.5yr, 9.5y, 7.25YR. The decimal branch is not optional:
    # without it "9.5y" masks only "5y", leaving "9." to be read as an amount.
    re.compile(r"\d+(?:\.\d+)?\s*-?\s*(?:y|yrs?|years?)\b", re.I),
    re.compile(r"\d+(?:\.\d+)?\s*nc\s*\d+(?:\.\d+)?", re.I),         # 15NC10, 6.2nc5.2
    re.compile(r"\bnc\s*\d+(?:\.\d+)?", re.I),                       # NC6.5
    re.compile(r"\b\d{2}s\b", re.I),                                 # 31s, 37s
    # capital-structure and tranche codes: AT1/RT1/T2/LT2/UT2/B3T2/L10Y and the
    # rest of the family, whether or not this extract happens to contain them
    re.compile(r"\b(?:L\d+Y|[LU]T\d|B\d?T\d|RT\d|AT\d|T\d)\b", re.I),
    re.compile(r"\d+\s*tranches?\b", re.I),                          # "the 6 tranches"
    re.compile(r"\b(?:" + MONTHS + r")[-/\s]?\d{2,4}\b", re.I),      # date maturities
    re.compile(r"\b(?:MS|T|SOFR|CT\d*|EFP)\s*\+\s*\d+(?:\.\d+)?"
               r"(?:\s*-\s*\d+)?\s*(?:bps?)?", re.I),                # spreads
    re.compile(r"\d+(?:\.\d+)?\s*%"),                                # coupons, ratios
    re.compile(r"\d+(?:\.\d+)?\s*x\b", re.I),                        # 2.0x
    re.compile(r"\d+\s*#|#\s*\d+"),                                  # 70#, #270
    re.compile(r"\b\d{1,2}[:.]?\d{0,2}\s*(?:AM|PM)\b", re.I),        # clock times
    re.compile(r"\d+\s*(?:accounts|accts|investors|orders|lines)\b", re.I),
    re.compile(r"\bDBR\b[^,;)|]*", re.I),                            # benchmark refs
    re.compile(r"\bBOOKBUILDING\b[^|]*\b(?:19|20)\d{2}\b[^|]*", re.I),
]

# lead-manager / prop demand — masked regardless of bracket position (§5.3)
LEAD_NOISE_PATTERNS = [
    # a bracketed group containing a lead keyword or an incl./excl. qualifier
    re.compile(r"\([^)]*(?:" + LEAD_KEYWORDS +
               r"|incl\.?|excl\.?|including|excluding|add[’'`]?l|additional)"
               r"[^)]*\)", re.I),
    # an unbracketed qualifier running to the end of its clause
    re.compile(r"(?:incl\.?|excl\.?|including|excluding|add[’'`]?l|additional"
               r"|\binc\b|\bex\b)[^,;)|]*", re.I),
    # "350m JLM interest" — the amount sits before the keyword
    re.compile(r"(?:" + CUR_RE + r")?\s*\d[\d,.]*\s*[A-Za-z]{0,4}\s*"
               r"(?:" + LEAD_KEYWORDS + r")\b[^,;)|]*", re.I),
    # a bare trailing keyword clause
    re.compile(r"\b(?:" + LEAD_KEYWORDS + r")\b[^,;)|]*", re.I),
]

# per-tranche breakdowns (§5.3). "interest split:" is a bond-level label, not a
# breakdown, and must never be masked.
BREAKDOWN_PATTERNS = [
    re.compile(r"\([^)]*(?<!interest )\bsplit\b[^)]*\)", re.I),
    re.compile(r"(?<!interest )\bsplit\s*:?\s*[\d€£$A-Z][^|)]*", re.I),
]

ENUM_SEPARATOR = re.compile(r"^[\s,/&+]*(?:and|&|/|,)?[\s,/&+]*$", re.I)


def normalise_currency(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return CURRENCY_ALIASES.get(raw.upper().replace("US$", "US$"), None) or \
        CURRENCY_ALIASES.get(raw, None) or \
        CURRENCY_ALIASES.get(raw.upper(), None)


# --------------------------------------------------------------------------
# §2  Data structures
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    position: int
    number: float
    currency: Optional[str]
    unit: Optional[str]

    def to_millions(self, unit: Optional[str] = None) -> Optional[float]:
        u = unit or self.unit
        if u is None:
            return None
        return self.number * UNIT_MULTIPLIER[u]


@dataclass
class Token:
    field_name: str            # "OUH" | "OIS"
    raw: str
    text: str
    masked: str = ""
    level: str = "BOND"        # "DEAL" | "BOND" | "BOTH"
    is_closing: bool = False
    amount: Optional[float] = None
    currency: Optional[str] = None
    unresolved: list[Candidate] = field(default_factory=list)
    enumeration_sum: Optional[float] = None

    def is_deal_level(self) -> bool:
        return self.level in ("DEAL", "BOTH")

    def is_bond_level(self) -> bool:
        return self.level in ("BOND", "BOTH")


@dataclass
class Row:
    source: dict[str, Any]
    deal_pk: str
    sub_deal_pk: str
    final: Optional[float]
    ouh_suppressed: bool = False
    ouh_drop_marked: bool = False
    tokens: list[Token] = field(default_factory=list)
    bond_peak: Value = NA
    bond_peak_ccy: Optional[str] = None
    bond_attr: Value = NA
    comments: list[str] = field(default_factory=list)
    comment_text: Optional[str] = None

    def add_comment(self, text: str) -> None:
        if text not in self.comments:
            self.comments.append(text)

    def has_bond_peak_comment(self) -> bool:
        return any(
            c.startswith("BOND:") and ("BOND_PEAK_VALUE" in c or "deal-level" in c
                                       or "no orderbook history" in c)
            for c in self.comments
        )


@dataclass
class Deal:
    deal_pk: str
    rows: list[Row]
    deal_peak: Value = NA
    deal_peak_ccy: Optional[str] = None
    deal_attr: Value = NA

    @property
    def is_multi(self) -> bool:
        return len(self.rows) > 1

    def add_comment_all(self, text: str) -> None:
        for row in self.rows:
            row.add_comment(text)


# --------------------------------------------------------------------------
# §1.2  Loading and §1.1 output
# --------------------------------------------------------------------------

def load_rows(path: str) -> list[Row]:
    text = open(path, encoding="utf-8").read()
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        # Elasticsearch triple-quoted values are not valid JSON
        text = re.sub(r'"""(.*?)"""', lambda m: json.dumps(m.group(1)), text,
                      flags=re.S)
        doc = json.loads(text)

    rows = []
    for hit in doc["hits"]["hits"]:
        src = hit["_source"]
        final = src.get("FINAL_ORDER_BOOK_SIZE")
        rows.append(Row(
            source=src,
            deal_pk=src.get("DEAL_PK"),
            sub_deal_pk=src.get("SUB_DEAL_PK"),
            final=float(final) if isinstance(final, (int, float)) else None,
        ))
    return rows


def group_by_deal(rows: Iterable[Row]) -> list[Deal]:
    groups: dict[str, list[Row]] = {}
    for row in rows:
        groups.setdefault(row.deal_pk, []).append(row)
    return [Deal(pk, rs) for pk, rs in groups.items()]


def build_output(deals: Iterable[Deal]) -> list[dict[str, Any]]:
    out = []
    for deal in deals:
        for row in deal.rows:
            obj = dict(row.source)
            obj["BOND_PEAK_VALUE"] = row.bond_peak
            obj["DEAL_PEAK_VALUE"] = deal.deal_peak
            obj["BOND_ATTRITION"] = row.bond_attr
            obj["DEAL_ATTRITION"] = deal.deal_attr
            obj["BOND_ATTRITION_COMMENTS"] = row.comment_text
            obj["ATTRITION"] = None
            out.append(obj)
    return out


def check_invariants(objects: list[dict[str, Any]]) -> list[str]:
    problems = []
    numeric_fields = ("BOND_PEAK_VALUE", "DEAL_PEAK_VALUE",
                      "BOND_ATTRITION", "DEAL_ATTRITION")
    by_deal: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        for f in numeric_fields:
            v = obj[f]
            if not (isinstance(v, (int, float)) or v == NA):
                problems.append(f"{obj['SUB_DEAL_PK']}: {f} is {v!r}")
        has_na = any(obj[f] == NA for f in numeric_fields)
        if has_na and obj["BOND_ATTRITION_COMMENTS"] is None:
            problems.append(f"{obj['SUB_DEAL_PK']}: N/A present but no comment")
        if not has_na and obj["BOND_ATTRITION_COMMENTS"] is not None:
            problems.append(f"{obj['SUB_DEAL_PK']}: comment without any N/A")
        by_deal.setdefault(obj["DEAL_PK"], []).append(obj)
    for pk, objs in by_deal.items():
        if len({str(o["DEAL_PEAK_VALUE"]) for o in objs}) > 1:
            problems.append(f"{pk}: DEAL_PEAK_VALUE differs across siblings")
        if len({str(o["DEAL_ATTRITION"]) for o in objs}) > 1:
            problems.append(f"{pk}: DEAL_ATTRITION differs across siblings")
    return problems


# --------------------------------------------------------------------------
# §2.1  Combined-in-OUH single-tranche
# --------------------------------------------------------------------------

def apply_combined_single_tranche(deal: Deal) -> None:
    if deal.is_multi:
        return
    row = deal.rows[0]
    ouh = row.source.get("ORDERBOOK_UPDATE_HISTORY") or ""
    ois = row.source.get("ORDERBOOK_INTEREST_SPLIT") or ""
    if ouh and DEAL_MARKERS.search(ouh):
        if ois.strip():
            row.ouh_suppressed = True
        else:
            row.ouh_drop_marked = True


# --------------------------------------------------------------------------
# §3  Tokenising and cleaning
# --------------------------------------------------------------------------

_DROPPED_PIPE = re.compile(r"(?<=>)(?=\s*[^<\s])")
_DASH_LABEL = re.compile(
    r"[-–,]\s*(?=(?:New\s+\d+\s*(?:y|yr|year)|FXD-FRN|FXD-to-FRN|FXD|FRN|FtF|Tap"
    r"|\d+\s*-?\s*(?:y|yr|yrs|year)s?|\d+NC\d+"
    r"|(?:" + MONTHS + r")[-\s]?\d{2,4})"
    r"[^>|]{0,15}>\s*(?:" + CUR_RE + r")?\s*\d)",
    re.I,
)


def split_tokens(text: str) -> list[str]:
    parts: list[str] = text.split("|")
    parts = [p for chunk in parts for p in _split_dropped_pipe(chunk)]
    parts = [p for chunk in parts for p in _DASH_LABEL.split(chunk)]
    return [p.strip() for p in parts if p and p.strip()]


def _split_dropped_pipe(chunk: str) -> list[str]:
    """Split after a <timestamp> when non-timestamp text follows it."""
    out, last = [], 0
    for m in TIMESTAMP.finditer(chunk):
        rest = chunk[m.end():]
        if rest.strip() and not rest.lstrip().startswith("<"):
            out.append(chunk[last:m.end()])
            last = m.end()
    out.append(chunk[last:])
    return out


def clean(piece: str) -> Optional[str]:
    t = TIMESTAMP.sub(" ", piece)              # 1
    t = t.replace("[", "").replace("]", "")    # 2 — keep the number
    t = re.sub(r"\s+", " ", t).strip()         # 3
    if not t:
        return None
    if JUNK_TOKEN.match(t):                    # 4 — pasted epoch
        return None
    if len(t) > BOILERPLATE_MAX_LEN or BOILERPLATE.search(t):   # 5
        return None
    return t                                   # 6 — round brackets kept


def tokenise_and_clean(row: Row) -> list[Token]:
    tokens: list[Token] = []
    fields = [("OUH", row.source.get("ORDERBOOK_UPDATE_HISTORY")),
              ("OIS", row.source.get("ORDERBOOK_INTEREST_SPLIT"))]
    for name, text in fields:
        if not text:
            continue
        if name == "OUH" and row.ouh_suppressed:
            continue
        for piece in split_tokens(text):
            cleaned = clean(piece)
            if cleaned is None:
                continue
            if name == "OUH" and row.ouh_drop_marked:
                # §2.1: keep only tokens with no deal marker and no closing keyword
                if DEAL_MARKERS.search(cleaned) or CLOSING.search(cleaned):
                    continue
            tokens.append(Token(field_name=name, raw=piece, text=cleaned))
    return tokens


# --------------------------------------------------------------------------
# §4  Classification
# --------------------------------------------------------------------------

def normalise_ouh(text: Optional[str]) -> str:
    if not text:
        return ""
    t = TIMESTAMP.sub(" ", text).lower()
    t = t.replace("[", "").replace("]", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def shared_ouh_keys(deal: Deal) -> set[str]:
    if not deal.is_multi:
        return set()
    counts: dict[str, int] = {}
    for row in deal.rows:
        key = normalise_ouh(row.source.get("ORDERBOOK_UPDATE_HISTORY"))
        if key:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n >= 2}


def classify_tokens(deal: Deal) -> None:
    shared = shared_ouh_keys(deal)
    for row in deal.rows:
        row_key = normalise_ouh(row.source.get("ORDERBOOK_UPDATE_HISTORY"))
        for tok in row.tokens:
            tok.is_closing = bool(CLOSING.search(tok.text))
            if DEAL_MARKERS.search(tok.text):
                tok.level = "DEAL"
            elif BOND_LABEL.search(tok.text):
                tok.level = "BOND"
            elif not deal.is_multi:
                tok.level = "BOTH"
            elif tok.field_name == "OIS":
                tok.level = "BOND"
            else:
                tok.level = "DEAL" if row_key in shared else "BOND"


# --------------------------------------------------------------------------
# §5  Amount extraction
# --------------------------------------------------------------------------

def _blank(match: re.Match) -> str:
    return " " * (match.end() - match.start())


def mask_noise(text: str, deal_level: bool) -> str:
    masked = text
    for pattern in NOISE_PATTERNS:
        masked = pattern.sub(_blank, masked)
    if deal_level:
        # §5.3: a per-tranche breakdown is noise only inside a DEAL-level token.
        # On a bond-level token "split 2.85+" is that bond's own figure.
        for pattern in BREAKDOWN_PATTERNS:
            masked = pattern.sub(_blank, masked)
    for pattern in LEAD_NOISE_PATTERNS:
        masked = pattern.sub(_blank, masked)
    return masked


_NUMBER = re.compile(
    r"(?P<cur>" + CUR_RE + r")?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*\+?\s*\.?\s*"
    r"(?P<tail>[A-Za-z€£¥$]*)"
)
_SLASH_PAIR = re.compile(r"[A-Z]{3}\s*/\s*[A-Z]{3}\s*$")


def _interpret_tail(tail: str) -> tuple[Optional[str], Optional[str]]:
    """Return (unit, currency-from-tail) for the alpha run after a number."""
    if not tail:
        return None, None
    low = tail.lower()
    # exact unit
    if low in UNIT_MULTIPLIER:
        return low, None
    # currency glued before the unit: "7.8 EURbn"
    for key in _CUR_KEYS:
        if low.startswith(key.lower()) and len(key) < len(tail):
            rest = tail[len(key):].lower()
            if rest in UNIT_MULTIPLIER:
                return rest, CURRENCY_ALIASES[key]
    # unit glued to something else. Try every prefix, longest first, and do NOT
    # stop at the first one that matches: "bnNOK" starts with the malformed unit
    # "bnn", whose remainder "ok" means nothing, but with "bn" the remainder is a
    # currency. Breaking early loses the whole amount.
    for unit in _UNIT_KEYS:
        if not low.startswith(unit):
            continue
        rest = tail[len(unit):]
        if not rest:
            return unit, None
        if rest.upper() in CURRENCY_ALIASES:              # "3.75bnNOK"
            return unit, CURRENCY_ALIASES[rest.upper()]
        if rest.lower().startswith(("final", "book")):    # "2.1bnFinal book"
            return unit, None
    return None, None


def find_amounts(masked: str) -> list[Candidate]:
    out: list[Candidate] = []
    for m in _NUMBER.finditer(masked):
        unit, tail_ccy = _interpret_tail(m.group("tail"))
        cur = normalise_currency(m.group("cur")) or tail_ccy
        prefix = masked[:m.start()]
        if _SLASH_PAIR.search(prefix.rstrip() + " ") or _SLASH_PAIR.search(prefix):
            cur = None            # §5.1 slash pair — take the number, drop the ccy
        out.append(Candidate(
            position=m.start("num"),
            number=float(m.group("num").replace(",", "")),
            currency=cur,
            unit=unit,
        ))
    return out


def _is_bare_enumeration(text: str, cands: list[Candidate], masked: str) -> bool:
    """A run of amounts with no headline in front of it (§5.4)."""
    if len(cands) < 2:
        return False
    if not re.search(r"respectively", text, re.I):
        return False
    for a, b in zip(cands, cands[1:]):
        between = masked[a.position:b.position]
        between = re.sub(r"^\s*[\d,.]+\s*[A-Za-z€£¥$]*", "", between)
        if not ENUM_SEPARATOR.match(between):
            return False
    return True


def select_headline(tok: Token, cands: list[Candidate], deal: Deal
                    ) -> tuple[Optional[float], Optional[str], list[Candidate]]:
    if not cands:
        return None, None, []

    if _is_bare_enumeration(tok.text, cands, tok.masked):
        if tok.is_deal_level() and len(cands) == len(deal.rows):
            unit = next((c.unit for c in cands if c.unit), None)
            if unit:
                tok.enumeration_sum = sum(c.to_millions(unit) for c in cands)
                ccy = next((c.currency for c in cands if c.currency), None)
                return None, ccy, []
        return None, None, []

    first = cands[0]
    if first.unit is not None:
        return first.to_millions(), first.currency, []
    return None, first.currency, [first]


def extract_amounts_pass1(deal: Deal) -> None:
    for row in deal.rows:
        for tok in row.tokens:
            tok.masked = mask_noise(tok.text, tok.is_deal_level())
            cands = find_amounts(tok.masked)
            amount, ccy, unresolved = select_headline(tok, cands, deal)
            tok.amount, tok.currency, tok.unresolved = amount, ccy, unresolved


OUTLIER_FACTOR = 10.0
OUTLIER_LOG: list[tuple[str, float, str]] = []


def apply_outlier_guard(deal: Deal) -> None:
    """Reject an amount more than 10x the largest other amount in the same row.

    Source typos such as `10yr >A$525bn` (beside `>A$500m`) and `>A$[4,090]bn`
    (beside `>A$[4,280]m`) parse correctly but are 1000x out. Reuses the
    factor-of-10 comparison from §5.2.

    Compare only within the same classification level: a DEAL-level combined
    figure is legitimately many times a single tranche's book, so measuring one
    against the other would reject real data (an 8-tranche deal's
    `Combined Books: €35.5bn+` beside per-tranche books of €2-3bn). A group with
    fewer than two amounts has nothing to compare against and is left alone.
    """
    for row in deal.rows:
        for level in ("DEAL", "BOND", "BOTH"):
            priced = [(t, t.amount) for t in row.tokens
                      if t.amount is not None and t.level == level]
            if len(priced) < 2:
                continue
            for tok, amount in priced:
                largest_other = max(a for t, a in priced if t is not tok)
                if largest_other > 0 and amount > OUTLIER_FACTOR * largest_other:
                    OUTLIER_LOG.append((row.sub_deal_pk, amount, tok.text[:80]))
                    tok.amount = None


def _within_factor_of_10(a: float, b: float) -> bool:
    if b == 0:
        return False
    ratio = a / b
    return 0.1 <= ratio <= 10


def _neighbour_amounts(tok: Token, row: Row, deal: Deal) -> list[float]:
    """Nearest first: same field, then the row's other field, then the deal."""
    same_field = [t for t in row.tokens
                  if t.field_name == tok.field_name and t is not tok]
    idx = row.tokens.index(tok)
    same_field.sort(key=lambda t: abs(row.tokens.index(t) - idx))
    other_field = [t for t in row.tokens if t.field_name != tok.field_name]
    other_rows = [t for r in deal.rows if r is not row for t in r.tokens]
    ordered = same_field + other_field + other_rows
    return [t.amount for t in ordered if t.amount is not None]


def extract_amounts_pass2(deal: Deal) -> None:
    any_anchor = any(t.amount is not None for r in deal.rows for t in r.tokens)
    for row in deal.rows:
        for tok in row.tokens:
            if tok.amount is not None or not tok.unresolved:
                continue
            cand = tok.unresolved[0]
            resolved = None
            for anchor in _neighbour_amounts(tok, row, deal):
                for unit in ("bn", "m"):
                    value = cand.to_millions(unit)
                    if _within_factor_of_10(value, anchor):
                        resolved = value
                        break
                if resolved is not None:
                    break
            if resolved is None and not any_anchor and row.final:
                for unit in ("bn", "m"):
                    value = cand.to_millions(unit)
                    if _within_factor_of_10(value, row.final):
                        resolved = value
                        break
            tok.amount = resolved


# --------------------------------------------------------------------------
# §6  Peaks
# --------------------------------------------------------------------------

def _pick_max(row: Row, tokens: list[Token]) -> None:
    best = max(tokens, key=lambda t: t.amount)
    row.bond_peak = best.amount
    row.bond_peak_ccy = best.currency


def compute_bond_peak(row: Row, deal: Deal) -> None:
    ois_bond = [t for t in row.tokens
                if t.field_name == "OIS" and t.is_bond_level()
                and not t.is_closing and t.amount is not None]
    if ois_bond:
        _pick_max(row, ois_bond)
        return

    ois_all = [t for t in row.tokens if t.field_name == "OIS"]
    ois_unusable = (not ois_all
                    or all(t.amount is None for t in ois_all)
                    or all(t.level == "DEAL" for t in ois_all))

    if ois_unusable:
        ouh_bond = [t for t in row.tokens
                    if t.field_name == "OUH" and t.is_bond_level()
                    and not t.is_closing and t.amount is not None]
        if ouh_bond:
            _pick_max(row, ouh_bond)
            return

    row.bond_peak = NA
    ouh_tokens = [t for t in row.tokens if t.field_name == "OUH"]
    if not row.tokens:
        row.add_comment("BOND: no orderbook history")
    elif deal.is_multi and ouh_tokens and all(t.level == "DEAL" for t in ouh_tokens):
        row.add_comment("BOND: only a deal-level figure available")
    else:
        row.add_comment("BOND: BOND_PEAK_VALUE missing")


def compute_deal_peak(deal: Deal) -> None:
    if not deal.is_multi:
        row = deal.rows[0]
        deal.deal_peak = row.bond_peak
        deal.deal_peak_ccy = row.bond_peak_ccy
        if deal.deal_peak == NA:
            deal.add_comment_all("DEAL: DEAL_PEAK_VALUE missing")
        return

    peaks = [r.bond_peak for r in deal.rows]
    ccys = {r.bond_peak_ccy for r in deal.rows if r.bond_peak_ccy is not None}

    if all(isinstance(p, float) for p in peaks):
        if len(ccys) <= 1:
            deal.deal_peak = sum(peaks)                     # PRIMARY rule (§6.2)
            deal.deal_peak_ccy = next(iter(ccys), None)
            return
        deal.deal_peak = NA
        deal.add_comment_all("DEAL: mixed-currency tranches, peak not summable")
        return

    deal_tokens = [t for r in deal.rows for t in r.tokens
                   if t.is_deal_level() and not t.is_closing
                   and (t.amount is not None or t.enumeration_sum is not None)]
    if deal_tokens:
        best = max(deal_tokens,
                   key=lambda t: t.amount if t.amount is not None
                   else t.enumeration_sum)
        deal.deal_peak = (best.amount if best.amount is not None
                          else best.enumeration_sum)
        deal.deal_peak_ccy = best.currency
    else:
        deal.deal_peak = NA
        deal.add_comment_all("DEAL: DEAL_PEAK_VALUE missing")


# --------------------------------------------------------------------------
# §7  Attrition
# --------------------------------------------------------------------------

def _subtract(peak: Value, final: Optional[float], scope: str,
              rows: list[Row], peak_ccy: Optional[str],
              final_ccy: Optional[str]) -> Value:
    peak_name = "BOND_PEAK_VALUE" if scope == "BOND" else "DEAL_PEAK_VALUE"

    def note(text: str) -> None:
        for row in rows:
            row.add_comment(text)

    if peak == NA:
        if scope == "BOND":
            if not rows[0].has_bond_peak_comment():
                note(f"{scope}: {peak_name} missing")
        else:
            note(f"{scope}: {peak_name} missing")
        return NA
    if final is None:
        note(f"{scope}: FINAL_ORDER_BOOK_SIZE missing")
        return NA
    if peak_ccy is not None and final_ccy is not None and peak_ccy != final_ccy:
        note("BOND: peak and final in different currencies")
        return NA
    if peak == final:
        note(f"{scope}: {peak_name} == FINAL_ORDER_BOOK_SIZE")
        return NA
    if peak < final:
        note(f"{scope}: {peak_name} < FINAL_ORDER_BOOK_SIZE")
        return NA
    return peak - final


def compute_attrition(deal: Deal) -> None:
    for row in deal.rows:
        row.bond_attr = _subtract(row.bond_peak, row.final, "BOND", [row],
                                  row.bond_peak_ccy, row.bond_peak_ccy)

    if deal.is_multi:
        if any(r.final is None for r in deal.rows):
            deal.deal_attr = NA
            deal.add_comment_all(
                "DEAL: FINAL_ORDER_BOOK_SIZE missing for one or more tranches")
            return
        f_deal = sum(r.final for r in deal.rows)
    else:
        f_deal = deal.rows[0].final

    deal.deal_attr = _subtract(deal.deal_peak, f_deal, "DEAL", deal.rows,
                               deal.deal_peak_ccy, deal.deal_peak_ccy)


# --------------------------------------------------------------------------
# §8  Comments
# --------------------------------------------------------------------------

def assemble_comments(deal: Deal) -> None:
    for row in deal.rows:
        bond = [c for c in row.comments if c.startswith("BOND:")]
        deal_c = [c for c in row.comments if c.startswith("DEAL:")]
        ordered = list(dict.fromkeys(bond + deal_c))
        row.comment_text = "; ".join(ordered) if ordered else None


# --------------------------------------------------------------------------
# §3 pipeline
# --------------------------------------------------------------------------

def process(deals: list[Deal]) -> None:
    for deal in deals:
        apply_combined_single_tranche(deal)
        for row in deal.rows:
            row.tokens = tokenise_and_clean(row)
        classify_tokens(deal)
        extract_amounts_pass1(deal)
        apply_outlier_guard(deal)      # before pass 2: a bogus anchor corrupts
        extract_amounts_pass2(deal)    # the inheritance in §5.2
        apply_outlier_guard(deal)
        for row in deal.rows:
            compute_bond_peak(row, deal)
        compute_deal_peak(deal)
        compute_attrition(deal)
        assemble_comments(deal)


def run(input_path: str) -> list[dict[str, Any]]:
    deals = group_by_deal(load_rows(input_path))
    process(deals)
    return build_output(deals)


# --------------------------------------------------------------------------
# §11 verification cases
# --------------------------------------------------------------------------

EQ = "BOND: BOND_PEAK_VALUE == FINAL_ORDER_BOOK_SIZE; " \
     "DEAL: DEAL_PEAK_VALUE == FINAL_ORDER_BOOK_SIZE"

CASES = [
    ("A", "UD0000000039509", {"SD0000000044790":
        dict(bond_peak=1450, bond_attr=600, comments=None)},
        dict(deal_peak=1450, deal_attr=600)),
    ("B", "UD0000000039022", {
        "SD0000000044184": dict(bond_peak=2850, bond_attr=1050, comments=None),
        "SD0000000044185": dict(bond_peak=2500, bond_attr=1200, comments=None),
        "SD0000000044186": dict(bond_peak=2550, bond_attr=1300, comments=None)},
        dict(deal_peak=7900, deal_attr=3550)),
    ("C", "UD0000000047117", {
        "SD0000000053835": dict(bond_peak=3400, bond_attr=200, comments=None),
        "SD0000000053836": dict(bond_peak=4300, bond_attr=400, comments=None)},
        dict(deal_peak=7700, deal_attr=600)),
    ("D", "UD0000000042921", {
        "SD0000000048922": dict(bond_peak=952, bond_attr=NA, comments=EQ),
        "SD0000000048923": dict(bond_peak=2650, bond_attr=NA, comments=EQ)},
        dict(deal_peak=3602, deal_attr=NA)),
    ("E", "UD0000000041756", {"SD0000000047495":
        dict(bond_peak=3900, bond_attr=1100, comments=None)},
        dict(deal_peak=3900, deal_attr=1100)),
    ("F", "UD0000000038557", {
        "SD0000000043651": dict(bond_peak=3000, bond_attr=600, comments=None),
        "SD0000000043652": dict(bond_peak=6000, bond_attr=2000, comments=None)},
        dict(deal_peak=9000, deal_attr=2600)),
    ("G", "UD0000000032177", {"SD0000000036271":
        dict(bond_peak=2900, bond_attr=1050, comments=None)},
        dict(deal_peak=2900, deal_attr=1050)),
    ("H", "UD0000000026854", {
        "SD0000000030173": dict(bond_peak=1100, bond_attr=NA),
        "SD0000000030315": dict(bond_peak=700, bond_attr=NA)},
        dict(deal_peak=1800, deal_attr=NA)),
]

FIELD_MAP = {"bond_peak": "BOND_PEAK_VALUE", "bond_attr": "BOND_ATTRITION",
             "deal_peak": "DEAL_PEAK_VALUE", "deal_attr": "DEAL_ATTRITION",
             "comments": "BOND_ATTRITION_COMMENTS"}


# Alpha runs that follow a number somewhere in the extracts and must NEVER be read
# as a unit. Bank names matter most: "BNP" begins with "bn", "BOSC" with "b".
NON_UNITS = [
    "yr", "y", "yrs", "year", "jlm", "jlms", "jbr", "mjlm", "nc", "fxd", "frn",
    "ftf", "tap", "fixed", "tranche", "tranches", "and", "of", "over", "at",
    "pre", "is", "accounts", "accts", "additional", "add", "excl", "orderbook",
    "total", "line", "respective", "am", "pm", "bps", "x", "nov", "cb", "ft",
    "bnp", "bosc", "bofa", "bank", "barclays", "ccb", "chf", "china", "dbs",
    "dongxing", "guotai", "hsbc", "mizuho", "sunriver", "agricultur", "eur",
    "books", "book", "bid", "bids", "min", "max",
]


def check_unit_collisions() -> list[str]:
    """No non-unit word may be interpretable as a unit (§5.1)."""
    bad = []
    for word in NON_UNITS:
        unit, _ = _interpret_tail(word)
        if unit is not None:
            bad.append(f"{word!r} is being read as unit {unit!r}")
    return bad


def run_tests(objects: list[dict[str, Any]]) -> int:
    by_sub = {o["SUB_DEAL_PK"]: o for o in objects}
    by_deal: dict[str, list[dict[str, Any]]] = {}
    for o in objects:
        by_deal.setdefault(o["DEAL_PK"], []).append(o)

    failures = 0
    for name, deal_pk, per_row, deal_exp in CASES:
        rows = by_deal.get(deal_pk)
        if not rows:
            print(f"  {name} {deal_pk}: SKIP (not in this extract)")
            continue
        errs = []
        for sub, expected in per_row.items():
            obj = by_sub.get(sub)
            if obj is None:
                errs.append(f"{sub} missing from output")
                continue
            for key, want in expected.items():
                got = obj[FIELD_MAP[key]]
                if isinstance(want, (int, float)) and isinstance(got, (int, float)):
                    ok = abs(got - want) < 0.01
                else:
                    ok = got == want
                if not ok:
                    errs.append(f"{sub}.{FIELD_MAP[key]}: got {got!r}, want {want!r}")
        for key, want in deal_exp.items():
            got = rows[0][FIELD_MAP[key]]
            if isinstance(want, (int, float)) and isinstance(got, (int, float)):
                ok = abs(got - want) < 0.01
            else:
                ok = got == want
            if not ok:
                errs.append(f"{FIELD_MAP[key]}: got {got!r}, want {want!r}")
        if errs:
            failures += 1
            print(f"  {name} {deal_pk}: FAIL")
            for e in errs:
                print(f"      {e}")
        else:
            print(f"  {name} {deal_pk}: pass")
    return failures


def summarise(objects: list[dict[str, Any]]) -> None:
    n = len(objects)
    def count(f):
        return sum(1 for o in objects if isinstance(o[f], (int, float)))
    print(f"rows: {n}")
    for f in ("BOND_PEAK_VALUE", "DEAL_PEAK_VALUE",
              "BOND_ATTRITION", "DEAL_ATTRITION"):
        c = count(f)
        print(f"  {f:<20} numeric {c:5}  N/A {n - c:5}")
    commented = sum(1 for o in objects if o["BOND_ATTRITION_COMMENTS"])
    print(f"  commented rows       {commented}")
    reasons: dict[str, int] = {}
    for o in objects:
        if o["BOND_ATTRITION_COMMENTS"]:
            for part in o["BOND_ATTRITION_COMMENTS"].split("; "):
                reasons[part] = reasons.get(part, 0) + 1
    for reason, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {c:5}  {reason}")
    if OUTLIER_LOG:
        print(f"  amounts rejected by the outlier guard: {len(OUTLIER_LOG)}")
        for sub, amount, text in OUTLIER_LOG:
            print(f"      {sub}  {amount:,.0f}  {text}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--test", action="store_true",
                    help="run the §11 verification cases")
    ap.add_argument("--summary", action="store_true",
                    help="print output statistics")
    args = ap.parse_args(argv)

    objects = run(args.input)

    problems = check_invariants(objects)
    if problems:
        print(f"invariant violations: {len(problems)}", file=sys.stderr)
        for p in problems[:20]:
            print(f"  {p}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(objects, fh, indent=2, ensure_ascii=False)
        print(f"wrote {len(objects)} rows to {args.output}")

    if args.summary:
        summarise(objects)

    if args.test:
        collisions = check_unit_collisions()
        print("unit-collision guard:", "pass" if not collisions
              else f"FAIL ({len(collisions)})")
        for c in collisions:
            print(f"      {c}")
        print("verification cases:")
        failures = run_tests(objects)
        return 1 if (failures or collisions) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
