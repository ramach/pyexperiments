from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

FIELD_ALIASES = {
    "invoice_id": ["Invoice Number", "Invoice No", "Invoice #", "Inv #"],
    "invoice_date": ["Invoice Date", "Date of Invoice", "Billing Date"],
    "due_date": ["Due Date", "Pay By", "Payment Due"],
    "vendor": ["Vendor", "Remit To", "From"],
    "client": ["Client", "Bill To", "To"],
    "total_amount": ["Total Amount", "Amount Due", "Total Due", "Balance Due"]
}

def sliding_windows(lines, max_window_size=3):
    for i in range(len(lines)):
        for window_size in range(2, max_window_size+1):
            if i + window_size <= len(lines):
                yield i, "\n".join(lines[i:i+window_size])

def semantic_field_match(text, field_aliases=FIELD_ALIASES, threshold=0.65, max_window=4):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    all_aliases = [(field, alias) for field, aliases in field_aliases.items() for alias in aliases]

    # Embed all aliases once
    alias_texts = [alias for (_, alias) in all_aliases]
    alias_embeddings = model.encode(alias_texts)

    matches = {}
    used_indices = set()

    for idx, window_text in sliding_windows(lines, max_window):
        window_embedding = model.encode([window_text])[0]
        sims = cosine_similarity([window_embedding], alias_embeddings)[0]
        max_sim_idx = sims.argmax()
        score = sims[max_sim_idx]
        if score >= threshold:
            field = all_aliases[max_sim_idx][0]
            if field not in matches:
                matches[field] = {
                    "value": extract_value_from_block(window_text),
                    "score": float(round(score, 3)),
                    "source": window_text
                }
                # mark lines as used
                for j in range(idx, idx+max_window):
                    used_indices.add(j)

    return {f: v["value"] for f, v in matches.items()}

def extract_value_from_block(block: str):
    """
    Return the likely value line (last non-label line) from a multi-line block.
    """
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if len(lines) < 2:
        return lines[0]
    return lines[-1]  # assume last line is value
