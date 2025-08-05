# job_match.py
import json
from typing import Dict, List, Any, Tuple, Optional
from pypdf import PdfReader
import re
import os
import time

import openai  # uses your existing OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

# --- Utilities ---
def read_pdf_text(file) -> str:
    """Read text from an uploaded PDF (Streamlit uploader file-like)."""
    try:
        reader = PdfReader(file)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

# --- LLM helpers ---
def gpt_json(prompt: str, retries: int = 2) -> dict:
    """
    Call OpenAI and force JSON output. We retry if the JSON fails to parse.
    """
    for _ in range(retries + 1):
        resp = openai.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a precise resume/job matching assistant that returns strict JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        try:
            return json.loads(text)
        except Exception:
            time.sleep(0.4)
    # Last resort
    return {"error": "Failed to parse JSON from model."}

# --- Core matching ---
MATCH_SYSTEM = """You evaluate how well a resume matches a job description.
Return STRICT JSON with this shape:
{
  "match_score": 0-100 integer,
  "top_reasons": [string, ...] (3-5 items),
  "missing_keywords": [string, ...] (max 12, deduplicate, lowercase),
  "nice_to_have": [string, ...] (0-10),
  "suggested_bullets": [string, ...] (2-6 STAR-style bullets, action-led, quantified if possible),
  "summary": "2-3 sentence summary for the candidate"
}
Scoring guidance:
- 80–100: Excellent fit,
- 60–79: Good but missing a few items,
- 40–59: Partial fit,
- <40: Low match.
Consider both hard skills (tools, languages) and soft/functional competencies.
"""

def analyze_match(resume_text: str, job_text: str) -> Dict[str, Any]:
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)

    if not resume_text or not job_text:
        return {"error": "Missing resume or job description text."}

    user_prompt = f"""{MATCH_SYSTEM}

RESUME:
\"\"\"{resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{job_text}\"\"\"
"""
    return gpt_json(user_prompt)
