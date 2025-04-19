# retrievers/serpapi_scraper.py
import os
from serpapi import GoogleSearch

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

def run_serpapi_scraper(query: str) -> str:
    if not SERPAPI_KEY:
        return "SERPAPI key is missing."

    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 3
    })

    results = search.get_dict()
    if "organic_results" not in results:
        return "No results found."

    top_snippets = [item["snippet"] for item in results["organic_results"] if "snippet" in item]
    return "\n".join(top_snippets)