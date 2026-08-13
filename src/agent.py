import json
from pathlib import Path

import pandas as pd
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parser import load_resumes


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

JOB_DESCRIPTION_PATH = BASE_DIR / "data" / "job_description.txt"
RESUME_FOLDER = BASE_DIR / "data" / "resumes"

OUTPUT_FOLDER = BASE_DIR / "output"

CSV_OUTPUT = OUTPUT_FOLDER / "ranked_candidates.csv"
JSON_OUTPUT = OUTPUT_FOLDER / "ranked_candidates.json"

MODEL_NAME = "llama3.2"


# ============================================================
# LOAD JOB DESCRIPTION
# ============================================================

def load_job_description():

    if not JOB_DESCRIPTION_PATH.exists():

        raise FileNotFoundError(
            f"Job description not found:\n"
            f"{JOB_DESCRIPTION_PATH}"
        )

    with open(
        JOB_DESCRIPTION_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    skills = [
        "python",
        "sql",
        "pandas",
        "numpy",
        "power bi",
        "tableau",
        "excel",
        "statistics",
        "data analysis",
        "machine learning",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "deep learning",
        "javascript",
        "java",
        "react",
        "mongodb",
        "flask",
        "git"
    ]

    text = text.lower()

    found_skills = []

    for skill in skills:

        if skill in text:

            found_skills.append(skill)

    return found_skills


# ============================================================
# SKILL MATCHING
# ============================================================

def calculate_skill_match(
    job_description,
    resume_text
):

    jd_skills = extract_skills(
        job_description
    )

    resume_skills = extract_skills(
        resume_text
    )

    if not jd_skills:

        return 0, [], []

    matched_skills = []

    missing_skills = []

    for skill in jd_skills:

        if skill in resume_skills:

            matched_skills.append(
                skill
            )

        else:

            missing_skills.append(
                skill
            )

    score = (
        len(matched_skills)
        /
        len(jd_skills)
    ) * 100

    return (
        round(score, 2),
        matched_skills,
        missing_skills
    )


# ============================================================
# NLP SIMILARITY
# ============================================================

def calculate_similarity(
    job_description,
    resume_text
):

    documents = [
        job_description,
        resume_text
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(
        float(similarity) * 100,
        2
    )


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_candidate(
    job_description,
    resume_text,
    similarity_score,
    skill_score,
    matched_skills,
    missing_skills
):

    prompt = f"""
You are an AI Resume Screening Agent.

Your job is to evaluate a candidate against a job description.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

NLP SIMILARITY SCORE:
{similarity_score}/100

SKILL MATCH SCORE:
{skill_score}/100

MATCHED SKILLS:
{", ".join(matched_skills)}

MISSING SKILLS:
{", ".join(missing_skills)}

Evaluate the candidate objectively.

IMPORTANT RULES:

1. Only use information that is actually present in the resume.
2. Never invent experience, education, skills or qualifications.
3. Missing information should be treated as a gap.
4. Explain both strengths and weaknesses.
5. Keep the answer concise.
6. Do not return JSON.
7. Follow the exact headings below.

Return your answer in this format:

RECOMMENDATION:
Strong Match

STRENGTHS:
- First strength
- Second strength
- Third strength

GAPS:
- First gap
- Second gap

REASONING:
Write 3 to 5 sentences explaining how well the candidate matches the job.
"""

    try:

        response = ollama.chat(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an objective and precise "
                        "AI recruitment screening assistant."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            options={
                "temperature": 0
            }
        )

        content = response[
            "message"
        ][
            "content"
        ].strip()

        return parse_ai_response(
            content
        )

    except Exception as error:

        print(
            f"   AI analysis failed: {error}"
        )

        return {

            "recommendation":
                "Needs Human Review",

            "strengths":
                "AI analysis failed.",

            "gaps":
                "Unable to determine.",

            "reasoning":
                str(error)
        }


# ============================================================
# PARSE AI RESPONSE
# ============================================================

def parse_ai_response(
    content
):

    recommendation = (
        "Needs Human Review"
    )

    strengths = ""

    gaps = ""

    reasoning = ""

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    upper_content = content.upper()

    if (
        "STRONG MATCH"
        in upper_content
    ):

        recommendation = "Strong Match"

    elif (
        "MODERATE MATCH"
        in upper_content
    ):

        recommendation = "Moderate Match"

    elif (
        "WEAK MATCH"
        in upper_content
    ):

        recommendation = "Weak Match"

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    if "STRENGTHS:" in content:

        strengths_section = content.split(
            "STRENGTHS:",
            1
        )[1]

        if "GAPS:" in strengths_section:

            strengths_section = (
                strengths_section.split(
                    "GAPS:",
                    1
                )[0]
            )

        strengths = (
            strengths_section
            .strip()
        )

    # --------------------------------------------------------
    # Gaps
    # --------------------------------------------------------

    if "GAPS:" in content:

        gaps_section = content.split(
            "GAPS:",
            1
        )[1]

        if "REASONING:" in gaps_section:

            gaps_section = (
                gaps_section.split(
                    "REASONING:",
                    1
                )[0]
            )

        gaps = (
            gaps_section
            .strip()
        )

    # --------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------

    if "REASONING:" in content:

        reasoning = (
            content.split(
                "REASONING:",
                1
            )[1]
            .strip()
        )

    return {

        "recommendation":
            recommendation,

        "strengths":
            strengths,

        "gaps":
            gaps,

        "reasoning":
            reasoning
    }


# ============================================================
# MAIN AGENT
# ============================================================

def run_agent():

    print()

    print("=" * 65)

    print(
        "                 AI RESUME SCREENING AGENT"
    )

    print("=" * 65)

    print()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "1. Loading Job Description..."
    )

    job_description = (
        load_job_description()
    )

    print(
        "   ✓ Job Description loaded"
    )

    print()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "2. Loading resumes..."
    )

    resumes = load_resumes(
        str(RESUME_FOLDER)
    )

    print()

    print(
        f"   ✓ {len(resumes)} resumes loaded"
    )

    print()

    if not resumes:

        print(
            "No resumes found."
        )

        print(
            f"Please add resumes to:\n"
            f"{RESUME_FOLDER}"
        )

        return

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "3. Analysing candidates..."
    )

    print()

    results = []

    for candidate_name, resume_text in (
        resumes.items()
    ):

        print(
            f"   → Analysing "
            f"{candidate_name}"
        )

        # --------------------------------------------
        # NLP SCORE
        # --------------------------------------------

        similarity_score = (
            calculate_similarity(
                job_description,
                resume_text
            )
        )

        # --------------------------------------------
        # SKILL SCORE
        # --------------------------------------------

        (
            skill_score,
            matched_skills,
            missing_skills
        ) = calculate_skill_match(
            job_description,
            resume_text
        )

        # --------------------------------------------
        # AI ANALYSIS
        # --------------------------------------------

        ai_result = analyze_candidate(

            job_description,

            resume_text,

            similarity_score,

            skill_score,

            matched_skills,

            missing_skills
        )

        # --------------------------------------------
        # FINAL SCORE
        # --------------------------------------------

        final_score = round(

            (
                similarity_score * 0.40
            )
            +
            (
                skill_score * 0.60
            ),

            2
        )

        results.append({

            "candidate":
                candidate_name,

            "final_score":
                final_score,

            "similarity_score":
                similarity_score,

            "skill_match":
                skill_score,

            "matched_skills":
                ", ".join(
                    matched_skills
                ),

            "missing_skills":
                ", ".join(
                    missing_skills
                ),

            "recommendation":
                ai_result[
                    "recommendation"
                ],

            "strengths":
                ai_result[
                    "strengths"
                ],

            "gaps":
                ai_result[
                    "gaps"
                ],

            "reasoning":
                ai_result[
                    "reasoning"
                ]
        })

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    results.sort(

        key=lambda candidate:
            candidate[
                "final_score"
            ],

        reverse=True
    )

    # Add ranking

    for rank, candidate in enumerate(
        results,
        start=1
    ):

        candidate[
            "rank"
        ] = rank

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print()

    print("=" * 65)

    print(
        "                       FINAL RANKING"
    )

    print("=" * 65)

    print()

    for candidate in results:

        print(
            f"#{candidate['rank']} "
            f"{candidate['candidate']}"
        )

        print(
            f"   Final Score: "
            f"{candidate['final_score']}/100"
        )

        print(
            f"   NLP Similarity: "
            f"{candidate['similarity_score']}/100"
        )

        print(
            f"   Skill Match: "
            f"{candidate['skill_match']}/100"
        )

        print(
            f"   Recommendation: "
            f"{candidate['recommendation']}"
        )

        print()

        print(
            "   Strengths:"
        )

        print(
            candidate["strengths"]
        )

        print()

        print(
            "   Gaps:"
        )

        print(
            candidate["gaps"]
        )

        print()

        print(
            "   Reasoning:"
        )

        print(
            candidate["reasoning"]
        )

        print()

        print("-" * 65)

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = pd.DataFrame(
        results
    )

    columns = [

        "rank",

        "candidate",

        "final_score",

        "similarity_score",

        "skill_match",

        "matched_skills",

        "missing_skills",

        "recommendation",

        "strengths",

        "gaps",

        "reasoning"
    ]

    dataframe = dataframe[
        columns
    ]

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    dataframe.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()

    print("=" * 65)

    print(
        "                         COMPLETE"
    )

    print("=" * 65)

    print()

    print(
        f"CSV saved to:\n"
        f"{CSV_OUTPUT}"
    )

    print()

    print(
        f"JSON saved to:\n"
        f"{JSON_OUTPUT}"
    )

    print()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    run_agent()