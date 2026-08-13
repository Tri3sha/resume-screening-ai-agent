@'
# AI Resume Screening Agent

An AI-powered agent that screens and ranks resumes against a Job Description using NLP, skill matching, and a local LLM.

## Features

- PDF resume parsing
- TF-IDF NLP similarity
- Skill matching
- Llama 3.2 AI evaluation through Ollama
- Candidate strengths and gaps
- Ranked candidates
- CSV and JSON output
- Supports 10+ resumes in one run

## Architecture

Job Description + Resumes
        ↓
PDF Parser
        ↓
TF-IDF + Skill Matching
        ↓
Llama 3.2
        ↓
Candidate Evaluation
        ↓
Ranked CSV + JSON

## Project Structure

resume-screening-ai-agent/
├── data/
│   ├── job_description.txt
│   └── resumes/
├── output/
├── src/
│   ├── agent.py
│   └── parser.py
├── .gitignore
└── README.md

## Setup

### 1. Clone

git clone https://github.com/Tri3sha/resume-screening-ai-agent.git
cd resume-screening-ai-agent

### 2. Create environment

python -m venv venv
venv\Scripts\activate

### 3. Install dependencies

pip install pandas scikit-learn ollama pymupdf

### 4. Install Ollama

Download Ollama:
https://ollama.com/download/windows

Then run:

ollama pull llama3.2

### 5. Run

python src/agent.py

## Scoring

Final Score = 40% NLP Similarity + 60% Skill Match

The LLM provides qualitative evaluation including strengths, gaps, recommendation, and reasoning.

## Output

Results are saved to:

output/ranked_candidates.csv
output/ranked_candidates.json

## Sample Dataset

The repository contains 12 sample resumes for batch screening.

## Tradeoffs

TF-IDF is fast and interpretable but has limited semantic understanding.

Rule-based skill matching is transparent but depends on the predefined skill vocabulary.

Llama 3.2 through Ollama runs locally and requires no external API credits, but smaller local models may provide less sophisticated reasoning.

## Limitations

This is a prototype for initial screening. Results should be reviewed by a human and should not be the sole basis for hiring decisions.

## Future Improvements

- Semantic embeddings
- Better skill extraction
- OCR support
- Web interface
- Automated testing
- Recruiter dashboard
- Bias and fairness evaluation

## Author

Trisha

GitHub: https://github.com/Tri3sha
'@ | Set-Content -Encoding UTF8 README.md

git add README.md
git commit -m "Add README documentation"
git push
