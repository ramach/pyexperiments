from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

FIELD_ALIASES = {
    "invoice_id": ["Invoice Number", "Invoice No", "Invoice #", "Inv #"],
    "invoice_date": ["Invoice Date", "Date of Invoice", "Billing Date"],
    "due_date": ["Due Date", "Pay By", "Payment Due"],
    "vendor": ["Vendor", "Remit To", "From"],
    "client": ["Client", "Bill To", "To"],
    "total_amount": ["Total Amount", "Amount Due", "Total Due", "Balance Due"]
}

def extract_lines(text: str) -> list:
    return [line.strip() for line in text.split("\n") if line.strip()]

def is_probably_another_label(line: str, all_aliases: set, threshold: float = 0.85) -> bool:
    """Check if a line is itself a known label."""
    line_embedding = model.encode([line])
    alias_embeddings = model.encode(list(all_aliases))
    sim = cosine_similarity(line_embedding, alias_embeddings)
    return sim.max() > threshold

def extract_value_from_next_non_label_line(lines, idx, all_aliases):
    for j in range(idx + 1, len(lines)):
        line = lines[j].strip()
        if not line:
            continue
        if is_probably_another_label(line, all_aliases):
            continue
        return line
    return ""

def match_fields(text: str, field_aliases=FIELD_ALIASES, threshold=0.7, return_scores=False) -> dict:
    lines = extract_lines(text)
    if not lines:
        return {}

    line_embeddings = model.encode(lines)
    results = {}
    all_aliases = set([alias for aliases in field_aliases.values() for alias in aliases])

    for field, aliases in field_aliases.items():
        alias_embeddings = model.encode(aliases)
        sims = cosine_similarity(alias_embeddings, line_embeddings)

        # Find the best label match
        best_score = -1
        best_idx = -1
        for i in range(len(lines)):
            score = max(sims[:, i])
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold:
            value = extract_value_from_next_non_label_line(lines, best_idx, all_aliases)
            if return_scores:
                results[field] = {"value": value, "score": float(round(best_score, 3))}
            else:
                results[field] = value

    return results
