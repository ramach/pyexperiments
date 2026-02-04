from __future__ import annotations

import random
import re
from dataclasses import dataclass
from collections import Counter
from typing import List, Optional, Tuple, Dict

import hashlib
from datetime import date as _date


VOWELS = set("aeiou")

def starts_with_vowel(w: str) -> bool:
    w = normalize_word(w)
    return bool(w) and w[0] in VOWELS

def contains_input_word_as_is(solution: str, w1: str, w2: str) -> bool:
    sol = normalize_word(solution)
    a = normalize_word(w1)
    b = normalize_word(w2)
    return (a and a in sol) or (b and b in sol)

# ---------------------------
# Phrase bank (v1 seed)
# Each entry: (word1, word2, phrase_label)
# Keep words lowercase (no punctuation). Phrase label is display-only.
# ---------------------------
PHRASES = [
    ("last", "straw", "the last straw"),
    ("rain", "delay", "rain delay"),
    ("like", "never", "like never before"),
    ("dead", "wrong", "dead wrong"),
    ("cold", "feet", "cold feet"),
    ("high", "noon", "high noon"),
    ("fast", "track", "fast track"),
    ("hard", "ball", "play hardball"),
    ("long", "shot", "long shot"),
    ("open", "secret", "open secret"),
    ("short", "fuse", "short fuse"),
    ("black", "sheep", "black sheep"),
    ("green", "light", "green light"),
    ("silver", "lining", "silver lining"),
    ("prime", "time", "prime time"),
    ("crash", "course", "crash course"),
    ("quick", "fix", "quick fix"),
    ("small", "talk", "small talk"),
    ("heavy", "metal", "heavy metal"),
    ("early", "bird", "early bird"),
    ("first", "hand", "firsthand"),
    ("final", "word", "final word"),
]

WORD_RE = re.compile(r"^[a-z]+$", re.I)

# ---------------------------
# English word validation
# - Uses wordfreq if available (recommended)
# - Fallback to small built-in list if not installed
# ---------------------------
FALLBACK_WORDS = {
    # tiny starter set (add more as needed)
    "straw", "last", "stall", "walls", "swat", "salt", "twas", "warts",
    "delay", "ready", "layer", "relay",
    "never", "lever", "nerve",
    "wrong", "grown", "word", "row",
    "feet", "fete",
    "lining", "silver",
    "prime", "time",
    "course", "crash",
    "metal", "heavy",
    "talk", "small",
    "green", "light",
    "black", "sheep",
}

def puzzle_id_from_words(word1: str, word2: str, len_a: int | None = None, len_b: int | None = None, day: _date | None = None) -> str:
    """
    Stable ID: based on sorted words + lengths + optional day (for daily puzzles).
    """
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)
    a, b = sorted([w1, w2])
    parts = [a, b]
    if len_a is not None and len_b is not None:
        parts.append(f"{int(len_a)}x{int(len_b)}")
    if day is not None:
        parts.append(day.isoformat())
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]  # short id

def difficulty_from_solution_count(n: int) -> str:
    """
    Very simple tiering (tune anytime).
    """
    if n <= 10:
        return "Expert"
    if n <= 50:
        return "Hard"
    if n <= 200:
        return "Medium"
    return "Easy"

def compute_difficulty(puzzle: Puzzle, min_consonant_len: int, max_words: int = 200_000) -> dict:
    """
    Computes: number of solutions + tier.
    Uses all_valid_solutions but asks for a higher limit to estimate count.
    """
    sols = all_valid_solutions(
        puzzle,
        min_consonant_len=min_consonant_len,
        limit=5000,          # cap display results but enough for count estimate
        max_words=max_words
    )
    n = len(sols)
    return {
        "solutions_found": n,
        "tier": difficulty_from_solution_count(n),
    }

def is_valid_english_word(w: str, min_zipf: float = 2.0) -> bool:
    """
    Returns True if 'w' looks like a valid English word.
    Uses wordfreq if installed; else fallback set.
    """
    w = w.strip().lower()
    if not w or not WORD_RE.match(w):
        return False

    try:
        from wordfreq import zipf_frequency  # type: ignore
        return zipf_frequency(w, "en") >= min_zipf
    except Exception:
        return w in FALLBACK_WORDS


# ---------------------------
# Core puzzle logic
# ---------------------------
@dataclass
class Puzzle:
    word1: str
    word2: str
    phrase_label: str

    @property
    def letters(self) -> Counter:
        return Counter((self.word1 + self.word2).lower())

    @property
    def total_len(self) -> int:
        return len(self.word1) + len(self.word2)

def normalize_word(w: str) -> str:
    w = (w or "").strip().lower()
    # keep letters only (v1)
    w = re.sub(r"[^a-z]", "", w)
    return w

def can_build_from_letters(candidate: str, letters: Counter) -> Tuple[bool, Dict[str, int]]:
    """
    Checks multiset containment: candidate letters must be <= available letters.
    Returns (ok, overuse_dict) where overuse_dict shows excess letters if not ok.
    """
    cand = normalize_word(candidate)
    need = Counter(cand)
    over = {}
    for ch, cnt in need.items():
        if cnt > letters.get(ch, 0):
            over[ch] = cnt - letters.get(ch, 0)
    return (len(over) == 0, over)

def pick_puzzle(len_a: int = 5, len_b: int = 4, allow_swap: bool = True) -> Puzzle:
    """
    Pick a puzzle where word lengths match (len_a,len_b) or swapped if allow_swap.
    """
    candidates = []
    for w1, w2, label in PHRASES:
        if (len(w1) == len_a and len(w2) == len_b) or (allow_swap and len(w1) == len_b and len(w2) == len_a):
            candidates.append((w1, w2, label))

    if not candidates:
        # fallback: pick any phrase
        w1, w2, label = random.choice(PHRASES)
        return Puzzle(w1, w2, label)

    w1, w2, label = random.choice(candidates)
    return Puzzle(w1, w2, label)

def grade_solution_old(
        puzzle: Puzzle,
        solution: str,
        min_letters_used: Optional[int] = None,
        require_english: bool = True,
) -> dict:
    """
    Returns a structured grading result for one solution word.
    """
    sol = normalize_word(solution)
    if not sol:
        return {"ok": False, "reason": "Empty solution."}

    if min_letters_used is None:
        min_letters_used = max(1, puzzle.total_len - 2)

    ok_letters, over = can_build_from_letters(sol, puzzle.letters)
    if not ok_letters:
        return {
            "ok": False,
            "reason": "Uses letters not available (or too many of a letter).",
            "overuse": over,
            "solution": sol,
        }

    if len(sol) < min_letters_used:
        return {
            "ok": False,
            "reason": f"Too short. Needs at least {min_letters_used} letters.",
            "solution": sol,
            "len": len(sol),
            "min_required": min_letters_used,
        }

    if require_english and not is_valid_english_word(sol):
        return {
            "ok": False,
            "reason": "Not recognized as an English word (dictionary check failed).",
            "solution": sol,
        }

    return {
        "ok": True,
        "solution": sol,
        "len": len(sol),
        "min_required": min_letters_used,
    }

def grade_bonus_phrase_old(solution: str, bonus_phrase: str) -> dict:
    """
    v1 bonus: check that the bonus phrase contains the solution word as a standalone-ish token.
    (We’ll make this smarter later: idiom database / phrase matcher.)
    """
    sol = normalize_word(solution)
    text = (bonus_phrase or "").strip().lower()
    # crude token containment
    tokens = re.findall(r"[a-z]+", text)
    ok = sol in tokens
    return {
        "ok": ok,
        "reason": "Contains solution word." if ok else "Does not contain solution word.",
        "tokens": tokens[:50],
    }

def explain_solution(puzzle: Puzzle, grade: dict) -> dict:
    """
    Produces a human-readable explanation for a graded solution.
    Assumes grade_solution() + apply_bonus_score() already ran.
    """
    if not grade.get("ok"):
        return {}

    sol = grade["solution"]
    letters_available = puzzle.letters
    letters_used = Counter(sol)

    remaining = letters_available - letters_used

    explanation = {
        "solution": sol,
        "starts_with_vowel": grade.get("starts_with_vowel", False),
        "length": len(sol),
        "min_required": grade.get("min_required"),
        "letters_used": dict(letters_used),
        "letters_remaining": dict(remaining),
        "base_score": grade.get("score_base"),
        "final_score": grade.get("score_final", grade.get("score_base")),
        "score_reason": [],
    }

    # Explain score logic
    if grade.get("score_final") == 95:
        explanation["score_reason"].append(
            "Bonus applied: solution used as a standalone word in a sentence."
        )
    else:
        if len(sol) >= 8:
            explanation["score_reason"].append("8+ letters used.")
        elif len(sol) == 7 and grade.get("starts_with_vowel"):
            explanation["score_reason"].append("7-letter word starting with a vowel.")
        elif len(sol) == 7:
            explanation["score_reason"].append("7-letter word starting with a consonant.")
        elif len(sol) == 6 and grade.get("starts_with_vowel"):
            explanation["score_reason"].append("6-letter word allowed due to vowel-start rule.")

    explanation["rule_checks"] = [
        "Used only available letters",
        "Did not contain input words as substrings",
        "Met minimum length rule",
        "Recognized as an English word",
    ]

    return explanation

import re

def grade_bonus_phrase_2(solution: str, bonus_phrase: str) -> dict:
    """
    v1.1 bonus: exact standalone word-boundary match for the solution.
    Accepts punctuation around the word, case-insensitive.
    Rejects substrings (e.g., 'unalready', 'alreadyish').
    """
    sol = normalize_word(solution)
    text = (bonus_phrase or "").strip()

    if not sol or not text:
        return {"ok": False, "reason": "No bonus phrase or solution provided."}

    # Word boundary match, case-insensitive:
    # \b works well with punctuation/spaces. We escape sol to be safe.
    pattern = re.compile(rf"\b{re.escape(sol)}\b", flags=re.IGNORECASE)

    ok = bool(pattern.search(text))
    return {
        "ok": ok,
        "reason": "Contains solution as a standalone word." if ok else "Does not contain solution as a standalone word.",
        "pattern": pattern.pattern,
    }

def apply_bonus_score(grade: dict, bonus_phrase: str) -> dict:
    """
    If grade is ok and bonus_phrase contains the solution token, upgrade score to 95.
    """
    if not grade.get("ok"):
        return grade
    sol = grade["solution"]
    b = grade_bonus_phrase(sol, bonus_phrase)
    grade = dict(grade)
    grade["bonus"] = b
    if bonus_phrase.strip() and b.get("ok"):
        grade["score_final"] = 95
    else:
        grade["score_final"] = grade.get("score_base", 0)
    return grade

def grade_solution(
        puzzle: Puzzle,
        solution: str,
        min_letters_used: Optional[int] = None,
        require_english: bool = True,
) -> dict:
    """
    Validates a solution and assigns a base score (no bonus sentence here).
    """
    sol = normalize_word(solution)
    if not sol:
        return {"ok": False, "reason": "Empty solution."}

    # NEW RULE: must not contain either input word as a substring
    if contains_input_word_as_is(sol, puzzle.word1, puzzle.word2):
        return {
            "ok": False,
            "reason": f"Solution contains an input word as-is ('{puzzle.word1}' or '{puzzle.word2}').",
            "solution": sol,
        }

    # Letter multiset check
    ok_letters, over = can_build_from_letters(sol, puzzle.letters)
    if not ok_letters:
        return {
            "ok": False,
            "reason": "Uses letters not available (or too many of a letter).",
            "overuse": over,
            "solution": sol,
        }

    total_len = puzzle.total_len
    default_min = max(1, total_len - 2)  # your default rule

    # NEW: vowel-start solutions allow one less char
    vowel = starts_with_vowel(sol)
    effective_min = default_min - 1 if vowel else default_min
    if min_letters_used is not None:
        # if caller provides a min, treat it as the consonant-min,
        # and apply vowel discount to that.
        effective_min = (min_letters_used - 1) if vowel else min_letters_used

    if len(sol) < effective_min:
        return {
            "ok": False,
            "reason": f"Too short. Needs at least {effective_min} letters"
                      + (" (vowel-start discount applied)." if vowel else "."),
            "solution": sol,
            "len": len(sol),
            "min_required": effective_min,
            "starts_with_vowel": vowel,
        }

    if require_english and not is_valid_english_word(sol):
        return {
            "ok": False,
            "reason": "Not recognized as an English word (dictionary check failed).",
            "solution": sol,
            "starts_with_vowel": vowel,
        }

    # ---- Scoring rules (base score, before bonus sentence) ----
    base = None
    if len(sol) >= 8:
        base = 90
    elif len(sol) == 7:
        base = 85 if vowel else 80
    elif len(sol) == 6 and vowel:
        base = 75
    else:
        # Any other valid case (e.g., smaller word lengths game modes)
        # score proportionally: 70 + 2*len, capped at 90
        base = min(90, 70 + 2 * len(sol))

    return {
        "ok": True,
        "solution": sol,
        "len": len(sol),
        "min_required": effective_min,
        "starts_with_vowel": vowel,
        "score_base": base,
    }

import re

def grade_bonus_phrase(solution: str, bonus_phrase: str) -> dict:
    """
    v1.1 bonus: exact standalone word-boundary match for the solution.
    Accepts punctuation around the word, case-insensitive.
    Rejects substrings (e.g., 'unalready', 'alreadyish').
    """
    sol = normalize_word(solution)
    text = (bonus_phrase or "").strip()

    if not sol or not text:
        return {"ok": False, "reason": "No bonus phrase or solution provided."}

    # Word boundary match, case-insensitive:
    # \b works well with punctuation/spaces. We escape sol to be safe.
    pattern = re.compile(rf"\b{re.escape(sol)}\b", flags=re.IGNORECASE)

    ok = bool(pattern.search(text))
    return {
        "ok": ok,
        "reason": "Contains solution as a standalone word." if ok else "Does not contain solution as a standalone word.",
        "pattern": pattern.pattern,
    }

import hashlib
from datetime import date as _date

def daily_seed(d: _date) -> int:
    s = d.isoformat().encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16)

def pick_daily_puzzle(d: _date, len_a: int = 5, len_b: int = 4, allow_swap: bool = True) -> Puzzle:
    rnd = random.Random(daily_seed(d))
    candidates = []
    for w1, w2, label in PHRASES:
        if (len(w1) == len_a and len(w2) == len_b) or (allow_swap and len(w1) == len_b and len(w2) == len_a):
            candidates.append((w1, w2, label))
    if not candidates:
        candidates = PHRASES[:]
    w1, w2, label = rnd.choice(candidates)
    return Puzzle(w1, w2, label)

def load_wordlist(max_n: int = 200_000) -> list[str]:
    """
    Uses wordfreq if available: top_n_list('en', N).
    Falls back to FALLBACK_WORDS.
    """
    try:
        from wordfreq import top_n_list  # type: ignore
        words = top_n_list("en", max_n)
        # keep only alpha, lowercase
        out = []
        for w in words:
            w = w.strip().lower()
            if WORD_RE.match(w):
                out.append(w)
        return out
    except Exception:
        return sorted(list(FALLBACK_WORDS))

def all_valid_solutions(
        puzzle: Puzzle,
        min_consonant_len: int | None = None,
        require_english: bool = True,   # keep True
        max_words: int = 200_000,
        limit: int = 2000
) -> list[dict]:
    """
    Returns a list of dicts: {solution, starts_with_vowel, len, score_base}
    Sorted by score then length.
    """
    wordlist = load_wordlist(max_words)

    total_len = puzzle.total_len
    base_min = max(1, total_len - 2) if min_consonant_len is None else min_consonant_len

    letters = puzzle.letters
    out = []

    for w in wordlist:
        # quick length pruning:
        vowel = starts_with_vowel(w)
        min_req = (base_min - 1) if vowel else base_min
        if len(w) < min_req:
            continue
        # must not contain input words
        if contains_input_word_as_is(w, puzzle.word1, puzzle.word2):
            continue
        # letter multiset check
        ok, _ = can_build_from_letters(w, letters)
        if not ok:
            continue
        # english validity already satisfied by wordlist; optionally keep require_english for fallback mode
        if require_english and not is_valid_english_word(w):
            continue

        # score using your rules (same as grade_solution)
        if len(w) >= 8:
            score = 90
        elif len(w) == 7:
            score = 85 if vowel else 80
        elif len(w) == 6 and vowel:
            score = 75
        else:
            score = min(90, 70 + 2 * len(w))

        out.append({"solution": w, "starts_with_vowel": vowel, "len": len(w), "score_base": score})

        if len(out) >= limit:
            # don’t keep growing indefinitely; we’ll sort and trim later
            pass

    # sort best-first
    out.sort(key=lambda x: (x["score_base"], x["len"], x["solution"]), reverse=True)
    return out[:limit]

def hint_from_solutions(solutions: list[dict]) -> dict:
    """
    Given candidate solutions (from all_valid_solutions), return a simple hint.
    """
    if not solutions:
        return {"type": "none", "text": "No hints available (no solutions found)."}
    best = solutions[0]["solution"]
    return {
        "type": "start_letter",
        "text": f"Hint: A strong solution starts with **'{best[0].upper()}'**.",
        "best_solution_preview": best[:2] + "…"  # keep spoiler-light
    }

import random

def build_hint_candidates(puzzle, min_consonant_len: int, require_english: bool, top_k: int = 100):
    """
    Returns a list of *validated* candidate solutions (dicts) that are safe to hint from.
    """
    # Get solver candidates (fast)
    sols = all_valid_solutions(puzzle, min_consonant_len=min_consonant_len, limit=top_k)

    # Re-validate with grade_solution so hint never points to something invalid due to rule drift
    validated = []
    for s in sols:
        r = grade_solution(
            puzzle=puzzle,
            solution=s["solution"],
            min_letters_used=min_consonant_len,
            require_english=require_english,
        )
        if r.get("ok"):
            validated.append(r)
    return validated

def make_hint(puzzle, validated_solutions: list[dict], hint_index: int = 0) -> dict:
    """
    Deterministic “rotating” hint: uses hint_index to cycle through hints.
    """
    if not validated_solutions:
        return {"type": "none", "text": "No hints available (no valid solutions found)."}

    # Pick a candidate from top N but rotate so it changes each click
    N = min(30, len(validated_solutions))
    pick = validated_solutions[hint_index % N]
    sol = pick["solution"]
    vowel = pick.get("starts_with_vowel", False)
    L = len(sol)

    # Build a menu of hint styles; rotate again for variety
    hint_styles = []

    # 1) broad category (non-spoilery)
    hint_styles.append({
        "type": "vowel_category",
        "text": f"Hint: A strong answer starts with a **{'vowel' if vowel else 'consonant'}**."
    })

    # 2) first letter (slightly more spoilery)
    hint_styles.append({
        "type": "first_letter",
        "text": f"Hint: Try starting with **'{sol[0].upper()}'**."
    })

    # 3) length suggestion (strategy)
    if L >= 8:
        hint_styles.append({"type": "length", "text": "Hint: Try for **8+ letters** for the 90-point tier."})
    elif L == 7:
        hint_styles.append({"type": "length", "text": "Hint: A **7-letter** solution is very plausible here."})
    elif L == 6 and vowel:
        hint_styles.append({"type": "length", "text": "Hint: A **6-letter** solution can work if it starts with a vowel."})

    # 4) “useful letter” hint (one letter from the solution) – relatively safe
    # Choose a letter that appears in the pool but is not too common (avoid 'a' if possible)
    letters = [ch for ch in sol if ch not in "ae"] or list(sol)
    ch = letters[(hint_index // 3) % len(letters)]
    hint_styles.append({"type": "letter", "text": f"Hint: Many good answers use the letter **'{ch.upper()}'**."})

    # Rotate hint style based on hint_index
    h = hint_styles[hint_index % len(hint_styles)]
    h["preview_len"] = L
    return h
