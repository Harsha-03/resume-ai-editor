import re
from typing import List, Tuple

def extract_keywords(text: str) -> List[str]:
    """
    Naive keyword extractor:
    - grabs capitalized tech terms / acronyms
    - grabs words >= 3 chars that look like skills
    You can replace with spaCy/KeyBERT later.
    """
    if not text:
        return []
    # techy tokens and acronyms
    tokens = re.findall(r"[A-Za-z][A-Za-z\+\#\.\-]{1,}", text)
    # de-duplicate, keep order
    seen = set()
    out = []
    for t in tokens:
        tt = t.strip().strip(",.();:")
        if len(tt) >= 3 and tt.lower() not in {"and","the","with","for","from","that","this","you","are","our","not","but","all"}:
            if tt.lower() not in seen:
                out.append(tt)
                seen.add(tt.lower())
    return out

def coverage(base: List[str], target: List[str]) -> Tuple[List[str], List[str]]:
    """Return (covered, missing) based on case-insensitive match with simple normalization."""
    bset = {b.lower() for b in base}
    covered, missing = [], []
    for t in target:
        if t.lower() in bset:
            covered.append(t)
        else:
            missing.append(t)
    return covered, missing
