import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import date
from typing import Optional

from leaderboard_db import init_db, submit_score, top_scores

from backend.game_backend import (
    pick_puzzle,
    pick_daily_puzzle,
    grade_solution,
    apply_bonus_score,
    explain_solution,
    all_valid_solutions,
    hint_from_solutions,
    puzzle_id_from_words,
    compute_difficulty,
)

st.set_page_config(page_title="Phrase-Forge", page_icon="🔤", layout="wide")
init_db()

# ----------------------------
# Session state defaults
# ----------------------------
if "puzzle" not in st.session_state:
    st.session_state.puzzle = pick_puzzle(5, 4, allow_swap=True)
if "history" not in st.session_state:
    st.session_state.history = []
if "player" not in st.session_state:
    st.session_state.player = "Player1"


# ----------------------------
# Sidebar settings (global)
# ----------------------------
with st.sidebar:
    st.header("Puzzle Settings")
    len_a = st.number_input("Word length A", min_value=2, max_value=12, value=5, step=1)
    len_b = st.number_input("Word length B", min_value=2, max_value=12, value=4, step=1)
    allow_swap = st.checkbox("Allow swap (4+5 or 5+4)", value=True)

    min_letters_mode = st.radio("Minimum solution length", ["total - 2 (default)", "custom"], index=0)
    if min_letters_mode == "custom":
        min_letters_custom = st.number_input(
            "Min letters in solution (consonant baseline)",
            min_value=1,
            max_value=24,
            value=int(len_a + len_b - 2),
            step=1,
        )
    else:
        min_letters_custom = None

    require_english = st.checkbox("Require English word (dictionary check)", value=True)
    st.caption("Tip: install `wordfreq` for a stronger dictionary and admin solver.")


# ----------------------------
# Tabs
# ----------------------------
tab_play, tab_leader, tab_admin = st.tabs(["🎮 Play", "🏆 Leaderboard", "🛠 Admin"])


# ======================================================
# TAB: PLAY
# ======================================================
with tab_play:
    st.title("🔤 Phrase-Forge")

    mode = st.radio("Mode", ["Daily", "Practice"], horizontal=True)

    # Player
    st.session_state.player = st.text_input("Player name", value=st.session_state.player)

    # Select puzzle
    if mode == "Daily":
        today = date.today()
        p = pick_daily_puzzle(today, int(len_a), int(len_b), allow_swap=allow_swap)
        st.session_state.puzzle = p
        day_key: Optional[str] = today.isoformat()
        st.caption(f"Daily puzzle for {today.isoformat()}")
    else:
        p = st.session_state.puzzle
        day_key = None

        if st.button("🎲 New practice puzzle"):
            st.session_state.puzzle = pick_puzzle(int(len_a), int(len_b), allow_swap=allow_swap)
            p = st.session_state.puzzle

    # Compute min length baseline (consonant baseline)
    total_len = len(p.word1) + len(p.word2)
    min_req_consonant = min_letters_custom if min_letters_custom is not None else max(1, total_len - 2)

    # Puzzle ID
    puzzle_id = puzzle_id_from_words(
        p.word1, p.word2,
        len_a=int(len_a), len_b=int(len_b),
        day=date.today() if mode == "Daily" else None
    )

    # Difficulty (cached)
    diff_key = f"diff::{puzzle_id}::{min_req_consonant}"
    if diff_key not in st.session_state:
        st.session_state[diff_key] = compute_difficulty(p, min_consonant_len=min_req_consonant)
    diff = st.session_state[diff_key]

    # Display puzzle header
    c1, c2, c3 = st.columns([1.2, 1.0, 1.3])
    with c1:
        st.subheader("Given words")
        st.markdown(f"## **{p.word1.upper()} {p.word2.upper()}**")
        st.caption(f"Phrase label: *{p.phrase_label}*")
        st.caption(f"Puzzle ID: `{puzzle_id}`")
    with c2:
        st.subheader("Constraints")
        st.write(f"Total letters: **{total_len}**")
        st.write(f"Min length baseline: **{min_req_consonant}** (vowel-start can be 1 less)")
        st.code("".join(sorted((p.word1 + p.word2).lower())), language="text")
    with c3:
        st.subheader("Difficulty")
        st.write(f"Tier: **{diff['tier']}**")
        st.write(f"Solutions found (cap): **{diff['solutions_found']}**")

    st.divider()

    # Word roles (player input; stored only in v2)
    st.subheader("Word roles (player input)")
    r1, r2 = st.columns(2)
    with r1:
        role1 = st.selectbox("Role of word 1", ["noun", "verb", "adjective", "adverb", "proper noun", "auxiliary", "other"])
    with r2:
        role2 = st.selectbox("Role of word 2", ["noun", "verb", "adjective", "adverb", "proper noun", "auxiliary", "other"])

    st.divider()

    # Hint button
    h1, h2 = st.columns([1, 3])
    with h1:
        if st.button("💡 Hint"):
            sols = all_valid_solutions(p, min_consonant_len=min_req_consonant, limit=200)
            hint = hint_from_solutions(sols)
            st.info(hint["text"])
    with h2:
        st.caption("Hint uses the solver (top word list). If slow, install `wordfreq` or reduce max_words in backend.")

    st.divider()

    # Solution submission
    st.subheader("Submit solutions")
    solutions_text = st.text_area(
        "Enter one solution per line",
        height=140,
        placeholder="e.g.\nalready\n... (one per line)",
    )
    bonus_phrase = st.text_input(
        "Bonus (optional): use your solution in a sentence/idiom (must contain exact word boundary)",
        value="",
    )

    # ----------------------------
    # This is the key part: populate results[]
    # ----------------------------
    results = []
    if st.button("✅ Grade"):
        lines = [ln.strip() for ln in (solutions_text or "").splitlines() if ln.strip()]

        if not lines:
            st.warning("Enter at least one solution word.")
        else:
            # Build results[] by grading each submitted line
            for sol in lines:
                r = grade_solution(
                    puzzle=p,
                    solution=sol,
                    min_letters_used=min_req_consonant,    # consonant baseline; backend applies vowel discount
                    require_english=require_english,
                )
                r = apply_bonus_score(r, bonus_phrase)      # upgrades to 95 if bonus condition matches
                results.append(r)

            # Display grading results
            if any(r.get("ok") for r in results):
                st.success("Grading complete.")
            else:
                st.error("No valid solutions yet. See feedback below.")

            for r in results:
                if r.get("ok"):
                    score = r.get("score_final", r.get("score_base", 0))
                    vowel_tag = "🅅" if r.get("starts_with_vowel") else ""
                    st.markdown(f"### ✅ `{r['solution']}` — **Score: {score}** {vowel_tag}")

                    with st.expander("Why this score?"):
                        exp = explain_solution(p, r)
                        st.markdown("**Score breakdown**")
                        for reason in exp.get("score_reason", []):
                            st.write(f"- {reason}")

                        st.markdown("**Rule checks**")
                        for chk in exp.get("rule_checks", []):
                            st.write(f"✓ {chk}")

                        cA, cB = st.columns(2)
                        with cA:
                            st.markdown("**Letters used**")
                            st.json(exp.get("letters_used", {}))
                        with cB:
                            st.markdown("**Letters remaining**")
                            st.json(exp.get("letters_remaining", {}))

                else:
                    st.markdown(f"### ❌ `{r.get('solution','')}`")
                    st.write(r.get("reason", "Invalid."))
                    if "overuse" in r:
                        st.code(r["overuse"])

            # Save session history (optional)
            st.session_state.history.append({
                "words": f"{p.word1} {p.word2}",
                "roles": (role1, role2),
                "solutions": results,
                "bonus": bonus_phrase,
                "puzzle_id": puzzle_id,
                "difficulty": diff["tier"],
                "mode": mode,
            })

    # Submit best score to leaderboard (only if results exist and have a valid entry)
    valid = [r for r in results if r.get("ok")]
    if valid:
        best = max(valid, key=lambda x: x.get("score_final", x.get("score_base", 0)))
        best_score = int(best.get("score_final", best.get("score_base", 0)))
        best_solution = best["solution"]

        if st.button("🏁 Submit best score to leaderboard"):
            submit_score(
                day=day_key,
                puzzle_id=puzzle_id,
                words=f"{p.word1} {p.word2}",
                difficulty=diff["tier"],
                player=st.session_state.player,
                score=best_score,
                solution=best_solution,
            )
            st.success(f"Submitted: {st.session_state.player} — {best_score} — {best_solution}")

    with st.expander("📜 Session history"):
        if not st.session_state.history:
            st.write("No plays yet.")
        else:
            for h in reversed(st.session_state.history[-20:]):
                st.markdown(f"**{h['words']}** | {h['mode']} | {h['difficulty']} | id={h['puzzle_id']}")
                ok = [x["solution"] for x in h["solutions"] if x.get("ok")]
                st.write(f"✅ {ok}" if ok else "✅ []")
                st.write("---")


# ======================================================
# TAB: LEADERBOARD
# ======================================================
with tab_leader:
    st.header("🏆 Leaderboard")

    view = st.radio("View", ["Today (Daily)", "This Puzzle", "All-time"], horizontal=True)

    # Grab current puzzle_id from session (if available)
    current_p = st.session_state.get("puzzle")
    if current_p:
        current_pid = puzzle_id_from_words(
            current_p.word1, current_p.word2,
            len_a=int(len_a), len_b=int(len_b),
            day=date.today() if st.session_state.get("mode", "Daily") == "Daily" else None
        )
    else:
        current_pid = None

    if view == "Today (Daily)":
        rows = top_scores(day=date.today().isoformat(), limit=30)
    elif view == "This Puzzle" and current_pid:
        rows = top_scores(puzzle_id=current_pid, limit=30)
    else:
        rows = top_scores(limit=30)

    if not rows:
        st.info("No scores yet.")
    else:
        st.dataframe(rows, use_container_width=True)


# ======================================================
# TAB: ADMIN
# ======================================================
with tab_admin:
    st.header("🛠 Admin / Solver")

    admin_code = st.text_input("Admin code (optional)", type="password")
    is_admin = (admin_code == "letmein") or (admin_code == "")

    if not is_admin:
        st.warning("Enter admin code to use solver tools.")
        st.stop()

    p = st.session_state.puzzle
    total_len = len(p.word1) + len(p.word2)
    min_req_consonant = min_letters_custom if min_letters_custom is not None else max(1, total_len - 2)

    st.write(f"Current puzzle: **{p.word1} {p.word2}** (min baseline {min_req_consonant})")
    limit = st.slider("Max solutions to display", 50, 2000, 200, 50)

    if st.button("🧠 Generate solutions"):
        sols = all_valid_solutions(p, min_consonant_len=min_req_consonant, limit=limit)
        st.session_state["admin_solutions"] = sols
        st.success(f"Found {len(sols)} solutions (showing up to {limit}).")

    sols = st.session_state.get("admin_solutions", [])
    if sols:
        show_spoilers = st.checkbox("Show full solutions (spoilers)", value=False)
        if show_spoilers:
            st.dataframe(sols, use_container_width=True)
        else:
            masked = [{"solution": s["solution"][:2] + "…", "len": s["len"], "score_base": s["score_base"]} for s in sols[:50]]
            st.dataframe(masked, use_container_width=True)
