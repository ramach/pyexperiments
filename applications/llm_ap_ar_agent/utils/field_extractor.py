
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

FIELD_ALIASES = {
    "invoice_id": ["Invoice Number", "Invoice No", "Invoice #", "Inv #"],
    "invoice_date": ["Invoice Date", "Billing Date"],
    "due_date": ["Due Date", "Payment Due", "Pay By"],
    "vendor": ["Vendor", "Remit To", "From"],
    "client": ["Client", "Bill To"],
    "total_amount": ["Total Amount", "Balance Due", "Amount Due"]
}

def sliding_windows(lines, size=3):
    for i in range(len(lines) - size + 1):
        yield i, "\n".join(lines[i:i+size])

def local_field_mapper(text, threshold=0.7):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    alias_pairs = [(field, alias) for field, aliases in FIELD_ALIASES.items() for alias in aliases]
    alias_texts = [alias for _, alias in alias_pairs]
    alias_embeds = model.encode(alias_texts)

    results = {}
    for i, block in sliding_windows(lines, size=3):
        block_embed = model.encode([block])[0]
        sims = cosine_similarity([block_embed], alias_embeds)[0]
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score >= threshold:
            field = alias_pairs[best_idx][0]
            if field not in results or best_score > results[field]["score"]:
                value = block.split("\n")[-1].strip()
                results[field] = {"value": value, "score": float(round(best_score, 3))}

    return results

def gpt_extract_fields_from_text(text):
    prompt = f"""
You are an expert in document understanding. Given the following invoice text, extract these fields as JSON:

- invoice_id
- invoice_date
- due_date
- vendor
- client
- total_amount

Only return valid JSON. If something is missing, return null for that field.

\n\nInvoice Text:\n{text}\n\nJSON:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return json.loads(response["choices"][0]["message"]["content"])

def extract_label_value_block(text, field_aliases=FIELD_ALIASES, threshold=0.75):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    embeddings = model.encode(lines)

    alias_pairs = [(field, alias) for field, aliases in field_aliases.items() for alias in aliases]
    alias_texts = [alias for _, alias in alias_pairs]
    alias_embeds = model.encode(alias_texts)

    label_lines = []
    for i, line_embed in enumerate(embeddings):
        sims = cosine_similarity([line_embed], alias_embeds)[0]
        best_score = max(sims)
        if best_score > threshold:
            field = alias_pairs[sims.argmax()][0]
            label_lines.append((i, field))

    if not label_lines:
        return {}

    label_indices = [idx for idx, _ in label_lines]
    label_block_end = max(label_indices)
    value_block_start = label_block_end + 1
    value_block_end = value_block_start + len(label_lines)

    values = lines[value_block_start:value_block_end]
    result = {}
    for (label_idx, field), value in zip(label_lines, values):
        if not value.strip():
            continue
        sim_to_aliases = cosine_similarity(model.encode([value]), alias_embeds).max()
        if sim_to_aliases >= 0.8:
            continue
        result[field] = value.strip()

    return result

def extract_invoice_fields(text, fallback_to_gpt=True):
    result = extract_label_value_block(text)
    embed_results = local_field_mapper(text)

    for f in FIELD_ALIASES:
        if f not in result and f in embed_results:
            result[f] = embed_results[f]["value"]

    def is_bad(v):
        return not v or v.lower() in {"invoice #", "invoice date", "terms"} or len(v) <= 2

    missing_final = [f for f in FIELD_ALIASES if f not in result or is_bad(result[f])]

    if fallback_to_gpt and missing_final:
        gpt_results = gpt_extract_fields_from_text(text)
        for f in missing_final:
            if gpt_results.get(f):
                result[f] = gpt_results[f]

    return result
