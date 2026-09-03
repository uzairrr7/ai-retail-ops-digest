# AI-Powered Retail Operations Digest

A personal project exploring how to combine data engineering with LLM-generated
reports while preventing AI hallucination through automated fact-checking.

Turns raw retail data (transactions, staffing shifts, returns, stores) into a
weekly, per-store metrics table, then uses an LLM to generate a short digest —
with every numeric claim the LLM makes checked against the real computed data
before it's shown to a user.

**Pipeline:** Raw CSVs → Data Audit → Clean + Join → Weekly Metrics → AI Digest
(two approaches compared) → Programmatic Claim Verification → Holdout Test

## Why this project

LLMs can state incorrect numbers confidently. This project compares two ways
of generating an AI report — plain prompting vs. structured output that's
verified before display — and shows that the structured approach catches
unsupported claims before a human ever sees them, not after.

## Setup

1. Install dependencies:

   pip install -r requirements.txt

2. Create a `.env` file in this folder with:

   GEMINI_API_KEY=your_key_here

   Get a free key from https://aistudio.google.com/apikey

3. Open `notebook.ipynb` and run all cells.

## What's inside

Project/
├── notebook.ipynb          <- the full pipeline, run top to bottom
├── source_code/
│   ├── data_audit.py         data quality audit
│   ├── cleaning.py           cleaning, joining, weekly metrics
│   ├── llm_client.py         LLM API wrapper (Gemini)
│   ├── digest_approach1.py   Approach 1 — prompt-based grounding
│   ├── digest_approach2.py   Approach 2 — structured + verified generation
│   ├── verifier.py           claim verification function
│   └── test_verifier.py      required verification test cases
├── data/                    sample retail CSVs
└── requirements.txt

## Running the non-AI parts (no API key needed)

The audit, cleaning, and metrics table all run standalone:

   cd source_code
   python data_audit.py
   python cleaning.py
   python test_verifier.py