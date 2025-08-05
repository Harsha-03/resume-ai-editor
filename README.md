# Resume Editor Automation (AI) — MVP

A minimal, fast-to-run Streamlit app that:
- Parses your resume (PDF/DOCX/TXT)
- Ingests a job description
- Highlights keyword gaps (ATS-style)
- Rewrites bullets and summary to match the role
- Exports a tailored resume as DOCX

## 1) Quickstart

```bash
# 1) Create and activate a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Copy .env.example and set your key
cp .env.example .env
# then edit .env to set OPENAI_API_KEY=...

# 4) Run
streamlit run streamlit_app.py
```

## 2) How it works (MVP)
- Upload a resume (PDF/DOCX/TXT) + paste a job description.
- Click **Analyze Fit** to see coverage and missing keywords.
- Click **Rewrite Bullets** to get quantified, impact-focused bullets.
- Click **Generate Tailored Resume** to get a rebuilt DOCX you can download.

## 3) Notes
- You can also paste your API key in the sidebar if you prefer not to use .env.
- This is an MVP. See the **Next Steps** section in the app for ideas to extend it (Firebase auth, history, PDF export, LinkedIn import, etc.).
