import json
import openai

def map_rule_text_to_structured(rule_text: str) -> dict:
    prompt = f"""
You are a document analyst. Analyze the rule below and return structured JSON with:
- rule_id (string or null)
- description (plain summary)
- condition (if any)
- action (what to do)
- confidence (as percentage from 0–100)

Rule:
\"\"\"
{rule_text}
\"\"\"

JSON format only.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = response.choices[0].message["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"error": "Failed to parse JSON", "raw": content}
