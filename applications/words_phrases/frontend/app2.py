import streamlit as st

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Phrase-Forge Word Game", page_icon="🔤", layout="wide")

st.title("🔤 Phrase-Forge (v1)")

# --- Safe import so we don't get a blank page ---
try:
    from backend.game_backend import explain_solution, pick_puzzle, grade_solution, grade_bonus_phrase, Puzzle
    backend_ok = True
except Exception as e:
    backend_ok = False
    st.error("Backend import failed. Fix this first (missing file, wrong folder, or syntax error).")
    st.exception(e)

if not backend_ok:
    st.stop()

# ----------------------------
# Session state
# ----------------------------
if "puzzle" not in st.session_state:
    st.session_state.puzzle = pick_puzzle(5, 4, allow_swap=True)
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Puzzle Settings")
    len_a = st.number_input("Word length A", min_value=2, max_value=12, value=5, step=1)
    len_b = st.number_input("Word length B", min_value=2, max_value=12, value=4, step=1)
    allow_swap = st.checkbox("Allow swap (4+5 or 5+4)", value=True)

    min_letters_mode = st.radio("Minimum solution length", ["total - 2 (default)", "custom"], index=0)
    if min_letters_mode == "custom":
        min_letters = st.number_input("Min letters in solution", min_value=1, max_value=24,
                                      value=int(len_a + len_b - 2), step=1)
    else:
        min_letters = None

    require_english = st.checkbox("Require English word (dictionary check)", value=True)

    if st.button("🎲 New puzzle"):
        st.session_state.puzzle = pick_puzzle(int(len_a), int(len_b), allow_swap=allow_swap)

# ----------------------------
# Main UI
# ----------------------------
p: Puzzle = st.session_state.puzzle

st.write("Two words form a valid phrase/idiom-ish pair. Classify them and forge a new word using ONLY their letters.")

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    st.subheader("Given words")
    st.markdown(f"## **{p.word1.upper()} {p.word2.upper()}**")
    st.caption(f"Phrase label: *{p.phrase_label}*")

with c2:
    st.subheader("Constraints")
    total_len = len(p.word1) + len(p.word2)
    min_req = min_letters if min_letters is not None else max(1, total_len - 2)
    st.write(f"Total letters available: **{total_len}**")
    st.write(f"Minimum solution length: **{min_req}**")
    st.code("".join(sorted((p.word1 + p.word2).lower())), language="text")

with c3:
    st.subheader("Word roles (player input)")
    role1 = st.selectbox("Role of word 1", ["noun", "verb", "adjective", "adverb", "proper noun", "auxiliary", "other"])
    role2 = st.selectbox("Role of word 2", ["noun", "verb", "adjective", "adverb", "proper noun", "auxiliary", "other"])
    st.caption("v1: stored only. Later we can auto-tag and score accuracy.")

st.divider()

st.subheader("Submit solutions")
solutions_text = st.text_area("Enter one solution per line", height=140, placeholder="e.g.\nstandstill\nstall\n...")

bonus_phrase = st.text_input("Bonus (optional): use your best solution in a phrase", value="")

if st.button("✅ Grade"):
    lines = [ln.strip() for ln in (solutions_text or "").splitlines() if ln.strip()]
    if not lines:
        st.warning("Enter at least one solution word.")
    else:
        from backend.game_backend import apply_bonus_score

        results = []
        for sol in lines:
            r = grade_solution(p, sol, min_letters_used=min_req, require_english=require_english)
            # apply bonus per-solution (uses the same bonus_phrase field)
            r2 = apply_bonus_score(r, bonus_phrase)
            results.append(r2)

        if any(r.get("ok") for r in results):
            st.success("Grading complete.")
        else:
            st.error("No valid solutions yet.")

        for r in results:
            if r["ok"]:
                score = r.get("score_final", r.get("score_base"))
                vowel_tag = "🅅" if r.get("starts_with_vowel") else ""
                st.markdown(
                    f"### ✅ `{r['solution']}` — **Score: {score}** {vowel_tag}"
                )
                # 🔍 Explanation panel
                with st.expander("Why this score?"):
                    exp = explain_solution(p, r)
                    st.markdown("**Score breakdown**")
                    for reason in exp.get("score_reason", []):
                        st.write(f"- {reason}")
                    st.markdown("**Rule checks**")
                    for chk in exp.get("rule_checks", []):
                        st.write(f"✓ {chk}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Letters used**")
                        st.json(exp.get("letters_used", {}))
                    with c2:
                        st.markdown("**Letters remaining**")
                        st.json(exp.get("letters_remaining", {}))
            else:
                st.markdown(f"### ❌ `{r.get('solution','')}`")
                st.write(r["reason"])
                if "overuse" in r:
                    st.code(r["overuse"])

        best = next((r["solution"] for r in results if r.get("ok")), None)
        if best and bonus_phrase.strip():
            b = grade_bonus_phrase(best, bonus_phrase)
            st.info("🏅 Bonus accepted!" if b["ok"] else f"Bonus not accepted: {b['reason']}")

        st.session_state.history.append({
            "words": f"{p.word1} {p.word2}",
            "roles": (role1, role2),
            "solutions": results,
            "bonus": bonus_phrase,
        })

st.divider()
with st.expander("📜 History (this session)"):
    if not st.session_state.history:
        st.write("No plays yet.")
    else:
        for h in reversed(st.session_state.history):
            st.markdown(f"**{h['words']}** | roles={h['roles'][0]}/{h['roles'][1]}")
            ok = [r["solution"] for r in h["solutions"] if r.get("ok")]
            st.write(f"✅ {ok}" if ok else "✅ []")
            st.write("---")
