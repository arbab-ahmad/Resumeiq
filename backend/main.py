"""
ResumeIQ — AI Resume Analyzer v2.0
====================================
Backend: FastAPI + SentenceTransformers + FAISS + Groq AI

Features:
  1. Upload & extract text from PDF / DOCX resumes
  2. Compute cosine similarity match score (embeddings)
  3. RAG — retrieve relevant resume tips via FAISS
  4. AI suggestions via Groq (llama-3.3-70b-versatile)
  5. Rank multiple resumes (recruiter mode)
  6. Detect weak bullet points surgically
  7. Rewrite weak bullets without fabrication
  8. User approves/rejects each change individually
  9. Re-score after approved changes applied
 10. Generate downloadable professional PDF resume (pure Python via reportlab)

Author : ResumeIQ Project
Version: 2.0.0
"""

import os
import io
import json
import uuid
import numpy as np
import faiss
import fitz                              # PyMuPDF  — PDF text extraction

from groq import Groq                    # Groq SDK  — free LLM API

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from docx import Document as DocxDocument

# reportlab — pure Python PDF generation, no LibreOffice needed
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


# =============================================================================
# PATH SETUP
# =============================================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DATA_DIR     = os.path.join(ROOT_DIR, "data")
TEMP_DIR     = os.path.join(BASE_DIR,  "temp_resumes")

os.makedirs(TEMP_DIR, exist_ok=True)

for folder, name in [(FRONTEND_DIR, "frontend"), (DATA_DIR, "data")]:
    if not os.path.exists(folder):
        raise RuntimeError(
            f"\n❌ Required folder '{name}' not found at: {folder}\n"
            f"   Make sure you run uvicorn from inside the 'backend/' folder.\n"
            f"   Command: cd resumeiq/backend && uvicorn main:app --reload --port 8000"
        )


# =============================================================================
# APP & MIDDLEWARE
# =============================================================================

app = FastAPI(title="ResumeIQ", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# =============================================================================
# AI MODEL — SentenceTransformers
# =============================================================================

print("⏳ Loading sentence embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded!")


# =============================================================================
# GROQ CLIENT — key from environment variable (never hardcode)
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "\n❌ GROQ_API_KEY environment variable is not set!\n"
        "   Get a free key at: https://console.groq.com\n"
        "   Then run:\n"
        "     Windows CMD  : set GROQ_API_KEY=gsk_your_key\n"
        "     PowerShell   : $env:GROQ_API_KEY='gsk_your_key'\n"
        "     Mac / Linux  : export GROQ_API_KEY=gsk_your_key"
    )

groq_client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq AI ready!")


# =============================================================================
# RAG — Build FAISS index from resume_tips.txt
# =============================================================================

def build_rag_index():
    tips_path = os.path.join(DATA_DIR, "resume_tips.txt")
    if not os.path.exists(tips_path):
        print("⚠️  resume_tips.txt not found — RAG disabled")
        return [], None

    with open(tips_path, "r", encoding="utf-8") as f:
        tips = [line.strip() for line in f if line.strip()]

    print(f"📚 Building RAG index from {len(tips)} tips...")
    tip_embeddings = np.array(embedding_model.encode(tips)).astype("float32")
    index = faiss.IndexFlatIP(tip_embeddings.shape[1])
    faiss.normalize_L2(tip_embeddings)
    index.add(tip_embeddings)
    print(f"✅ RAG index built with {index.ntotal} tips")
    return tips, index


TIPS, FAISS_INDEX = build_rag_index()


# =============================================================================
# HELPERS — TEXT EXTRACTION
# =============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# =============================================================================
# HELPERS — EMBEDDINGS & SCORING
# =============================================================================

def get_embedding(text: str) -> np.ndarray:
    return embedding_model.encode([text])[0]


def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    score = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    return round(float(score) * 100, 2)


def retrieve_relevant_tips(resume_text: str, top_k: int = 4) -> list:
    if not FAISS_INDEX or not TIPS:
        return []
    query_vec = embedding_model.encode([resume_text]).astype("float32")
    faiss.normalize_L2(query_vec)
    _, indices = FAISS_INDEX.search(query_vec, top_k)
    return [TIPS[i] for i in indices[0] if i < len(TIPS)]


# =============================================================================
# HELPERS — GROQ CALLS
# =============================================================================

def clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip().rstrip("```").strip()


def generate_ai_suggestions(resume_text: str, job_description: str,
                              score: float, relevant_tips: list) -> str:
    tips_block = "\n".join(f"- {t}" for t in relevant_tips) if relevant_tips else "None."
    prompt = f"""You are an expert career coach and resume reviewer.

Match Score     : {score}%
Job Description : {job_description[:1500]}
Resume          : {resume_text[:1500]}
Expert Tips     : {tips_block}

Provide a structured response with:
1. A brief assessment of the match score (1-2 sentences)
2. Top 3 specific improvements the candidate should make
3. Key skills or keywords missing from the resume
4. One encouraging closing sentence

Keep the response concise, practical, and encouraging."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.7,
    )
    return response.choices[0].message.content


def detect_weak_bullets(resume_text: str, job_description: str) -> list:
    prompt = f"""You are an expert resume analyst.

Analyze the resume below bullet point by bullet point.

CLASSIFICATION RULES:
- STRONG   : Specific, uses strong action verbs, quantified where possible, relevant to JD
- MODERATE : Has some specifics but could be stronger or better aligned to JD
- WEAK     : Vague, passive voice, generic, no metrics, not aligned with JD

JOB DESCRIPTION:
{job_description[:1200]}

RESUME:
{resume_text[:2500]}

Return ONLY a valid JSON array. No markdown, no explanation. Format:
[
  {{
    "original" : "exact bullet point or sentence text here",
    "strength" : "WEAK",
    "reason"   : "why it is weak",
    "section"  : "section name e.g. Work Experience"
  }}
]

Include every bullet point and achievement sentence. Do not skip any."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.2,
    )
    return json.loads(clean_json(response.choices[0].message.content))


def rewrite_weak_bullets(weak_bullets: list, job_description: str,
                          resume_text: str) -> list:
    if not weak_bullets:
        return []

    bullets_text = "\n\n".join(
        f"BULLET {i+1}:\nOriginal : {b['original']}\nWeakness : {b['reason']}"
        for i, b in enumerate(weak_bullets)
    )

    prompt = f"""You are an expert resume coach rewriting weak bullet points.

ABSOLUTE RULES:
1. DO NOT add any skill, tool, technology, or certification not already in the resume
2. DO NOT invent any numbers or percentages
3. DO NOT add responsibilities not already clearly implied
4. DO NOT change the fundamental claim or meaning
5. DO use stronger action verbs: developed, implemented, optimised, designed, led, built
6. DO naturally include relevant JD keywords where they HONESTLY apply
7. DO make vague claims more specific using context already in the resume
8. DO remove filler words and passive voice

FULL RESUME (context):
{resume_text[:2000]}

JOB DESCRIPTION (for keyword alignment only):
{job_description[:1000]}

BULLETS TO REWRITE:
{bullets_text}

Return ONLY a valid JSON array. No markdown, no explanation. Format:
[
  {{
    "original"     : "exact original text",
    "rewritten"    : "improved version",
    "changes_made" : "specific changes made",
    "jd_alignment" : "which JD keyword this now aligns to"
  }}
]"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.3,
    )
    return json.loads(clean_json(response.choices[0].message.content))


def apply_approved_changes(resume_text: str, approved_changes: list) -> str:
    updated = resume_text
    for change in approved_changes:
        if change.get("approved", False):
            orig = change.get("original", "")
            rewr = change.get("rewritten", "")
            if orig and rewr and orig in updated:
                updated = updated.replace(orig, rewr, 1)
    return updated


def extract_resume_as_json(improved_text: str) -> dict:
    prompt = f"""Parse the resume below into structured JSON.
Extract all information exactly as written — do not change, add, or remove anything.

RESUME:
{improved_text[:3000]}

Return ONLY valid JSON. No markdown. No explanation. Use this exact structure:
{{
  "name"           : "Full Name",
  "title"          : "Professional Title",
  "contact": {{
    "email"    : "email or empty string",
    "phone"    : "phone or empty string",
    "linkedin" : "linkedin url or empty string",
    "github"   : "github url or empty string",
    "location" : "city or empty string"
  }},
  "summary"        : "professional summary paragraph or empty string",
  "experience": [
    {{
      "role"    : "Job Title",
      "company" : "Company Name",
      "dates"   : "Start - End",
      "bullets" : ["bullet 1", "bullet 2"]
    }}
  ],
  "skills"         : ["skill1", "skill2"],
  "education": [
    {{
      "degree"      : "Degree Name",
      "institution" : "University Name",
      "year"        : "Year or date range",
      "grade"       : "GPA or percentage or empty string"
    }}
  ],
  "certifications" : ["cert1", "cert2"],
  "projects": [
    {{
      "name"    : "Project Name",
      "dates"   : "Year or empty string",
      "bullets" : ["bullet 1"]
    }}
  ]
}}

Use empty list [] or empty string "" for any section that does not exist."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1,
    )
    return json.loads(clean_json(response.choices[0].message.content))


# =============================================================================
# PDF GENERATION — pure Python via reportlab (no LibreOffice needed)
# =============================================================================

DARK_BLUE  = colors.HexColor("#1F4E79")
MID_BLUE   = colors.HexColor("#2E75B6")
LIGHT_GRAY = colors.HexColor("#888888")
GRAY       = colors.HexColor("#444444")


def build_resume_pdf(resume_json: dict, output_path: str):
    """
    Build a professional resume PDF from structured JSON using reportlab.
    Pure Python — no LibreOffice, no external software required.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
    )

    # ── Paragraph styles ──────────────────────────────────────────
    s_name = ParagraphStyle("s_name",
        fontName="Helvetica-Bold", fontSize=22,
        textColor=DARK_BLUE, alignment=TA_CENTER, spaceAfter=10,
    )
    s_title = ParagraphStyle("s_title",
        fontName="Helvetica", fontSize=12,
        textColor=MID_BLUE, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=6,  
    )
    s_contact = ParagraphStyle("s_contact",
        fontName="Helvetica", fontSize=9,
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=10,
    )
    s_section = ParagraphStyle("s_section",
        fontName="Helvetica-Bold", fontSize=10,
        textColor=MID_BLUE, spaceBefore=14, spaceAfter=3,
    )
    s_body = ParagraphStyle("s_body",
        fontName="Helvetica", fontSize=10,
        textColor=GRAY, spaceAfter=4, leading=14,
    )
    s_job_title = ParagraphStyle("s_job_title",
        fontName="Helvetica-Bold", fontSize=10.5,
        textColor=DARK_BLUE, spaceBefore=8, spaceAfter=1,
    )
    s_company = ParagraphStyle("s_company",
        fontName="Helvetica", fontSize=10,
        textColor=MID_BLUE, spaceAfter=2,
    )
    s_dates = ParagraphStyle("s_dates",
        fontName="Helvetica-Oblique", fontSize=9,
        textColor=LIGHT_GRAY, spaceAfter=3,
    )
    s_bullet = ParagraphStyle("s_bullet",
        fontName="Helvetica", fontSize=10,
        textColor=GRAY, leftIndent=14,
        spaceAfter=2, leading=14,
    )
    s_skills = ParagraphStyle("s_skills",
        fontName="Helvetica", fontSize=10,
        textColor=GRAY, spaceAfter=4, leading=14,
    )

    def hr():
        return HRFlowable(
            width="100%", thickness=0.8,
            color=MID_BLUE, spaceAfter=4, spaceBefore=2,
        )

    def section_head(title):
        return [Paragraph(title.upper(), s_section), hr()]

    def add_bullets(items):
        return [Paragraph(f"• {b.strip()}", s_bullet)
                for b in items if b.strip()]

    # ── Build story ────────────────────────────────────────────────
    story = []

    # Name
    if resume_json.get("name"):
        story.append(Paragraph(resume_json["name"], s_name))
        story.append(Spacer(1, 6))   # ← add this line

        # Title
    if resume_json.get("title"):
        story.append(Paragraph(resume_json["title"], s_title))
        story.append(Spacer(1, 4))   # ← add this line

        # Contact
        contact = resume_json.get("contact", {})
        parts = [v for v in [
            contact.get("email"), contact.get("phone"),
            contact.get("linkedin"), contact.get("github"),
            contact.get("location"),
        ] if v and v.strip()]
        if parts:
            story.append(Paragraph("  |  ".join(parts), s_contact))

        story.append(HRFlowable(width="100%", thickness=1.5, color=DARK_BLUE, spaceAfter=8))

        # Summary
        if resume_json.get("summary"):
            story += section_head("Professional Summary")
            story.append(Paragraph(resume_json["summary"], s_body))
            story.append(Spacer(1, 4))

        # Experience
        if resume_json.get("experience"):
            story += section_head("Work Experience")
            for job in resume_json["experience"]:
                if job.get("role"):
                    story.append(Paragraph(job["role"], s_job_title))
                if job.get("company"):
                    story.append(Paragraph(job["company"], s_company))
                if job.get("dates"):
                    story.append(Paragraph(job["dates"], s_dates))
                story += add_bullets(job.get("bullets", []))
                story.append(Spacer(1, 4))

        # Projects
        if resume_json.get("projects"):
            story += section_head("Projects")
            for proj in resume_json["projects"]:
                if proj.get("name"):
                    story.append(Paragraph(proj["name"], s_job_title))
                if proj.get("dates"):
                    story.append(Paragraph(proj["dates"], s_dates))
                story += add_bullets(proj.get("bullets", []))
                story.append(Spacer(1, 4))

        # Skills
        if resume_json.get("skills"):
            story += section_head("Technical Skills")
            story.append(Paragraph("  •  ".join(resume_json["skills"]), s_skills))
            story.append(Spacer(1, 4))

        # Education
        if resume_json.get("education"):
            story += section_head("Education")
            for edu in resume_json["education"]:
                if edu.get("degree"):
                    story.append(Paragraph(edu["degree"], s_job_title))
                info = "  —  ".join(x for x in [edu.get("institution"), edu.get("year")] if x)
                if info:
                    story.append(Paragraph(info, s_company))
                if edu.get("grade"):
                    story.append(Paragraph(edu["grade"], s_dates))
                story.append(Spacer(1, 4))

        # Certifications
        if resume_json.get("certifications"):
            story += section_head("Certifications")
            story += add_bullets(resume_json["certifications"])

        doc.build(story)


# =============================================================================
# API ROUTES — CORE
# =============================================================================

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    file_bytes = await file.read()
    filename   = file.filename.lower()
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    if not text:
        raise HTTPException(status_code=400,
            detail="Could not extract text. The file may be a scanned image PDF.")
    return {"filename": file.filename, "extracted_text": text, "word_count": len(text.split())}


@app.post("/analyze")
async def analyze_resume(
    resume_text:     str = Form(...),
    job_description: str = Form(...),
):
    if not resume_text.strip() or not job_description.strip():
        raise HTTPException(status_code=400, detail="Resume and job description cannot be empty.")
    resume_vec    = get_embedding(resume_text)
    jd_vec        = get_embedding(job_description)
    score         = compute_similarity(resume_vec, jd_vec)
    relevant_tips = retrieve_relevant_tips(resume_text, top_k=4)
    try:
        suggestions = generate_ai_suggestions(resume_text, job_description, score, relevant_tips)
    except Exception as e:
        suggestions = f"Suggestions unavailable: {e}\n\nTips:\n" + "\n".join(relevant_tips)
    return {"score": score, "suggestions": suggestions, "retrieved_tips": relevant_tips}


@app.post("/rank-resumes")
async def rank_resumes(
    files:           List[UploadFile] = File(...),
    job_description: str              = Form(...),
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 resumes.")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 resumes allowed.")
    jd_vec     = get_embedding(job_description)
    candidates = []
    for file in files:
        file_bytes = await file.read()
        fname      = file.filename.lower()
        try:
            text = extract_text_from_pdf(file_bytes) if fname.endswith(".pdf") \
                   else extract_text_from_docx(file_bytes) if fname.endswith(".docx") \
                   else None
        except Exception:
            text = None
        if not text:
            continue
        score = compute_similarity(get_embedding(text), jd_vec)
        candidates.append({"filename": file.filename, "score": score,
                            "word_count": len(text.split()), "preview": text[:200] + "..."})
    if not candidates:
        raise HTTPException(status_code=400, detail="No valid resumes could be processed.")
    candidates.sort(key=lambda x: x["score"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    return {"candidates": candidates, "total": len(candidates)}


# =============================================================================
# API ROUTES — IMPROVEMENT PIPELINE
# =============================================================================

@app.post("/suggest-improvements")
async def suggest_improvements(
    resume_text:     str = Form(...),
    job_description: str = Form(...),
):
    if not resume_text.strip() or not job_description.strip():
        raise HTTPException(status_code=400, detail="Resume and job description cannot be empty.")
    resume_vec     = get_embedding(resume_text)
    jd_vec         = get_embedding(job_description)
    original_score = compute_similarity(resume_vec, jd_vec)
    try:
        all_bullets = detect_weak_bullets(resume_text, job_description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bullet detection failed: {e}")
    weak   = [b for b in all_bullets if b.get("strength") in ["WEAK", "MODERATE"]]
    strong = [b for b in all_bullets if b.get("strength") == "STRONG"]
    try:
        suggested_changes = rewrite_weak_bullets(weak, job_description, resume_text) if weak else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rewriting failed: {e}")
    return {
        "original_score"   : original_score,
        "total_bullets"    : len(all_bullets),
        "strong_count"     : len(strong),
        "weak_count"       : len(weak),
        "suggested_changes": suggested_changes,
        "strong_bullets"   : [b["original"] for b in strong],
    }


@app.post("/apply-improvements")
async def apply_improvements(
    resume_text:      str = Form(...),
    job_description:  str = Form(...),
    approved_changes: str = Form(...),
):
    try:
        changes = json.loads(approved_changes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid approved_changes JSON.")
    updated        = apply_approved_changes(resume_text, changes)
    jd_vec         = get_embedding(job_description)
    original_score = compute_similarity(get_embedding(resume_text), jd_vec)
    improved_score = compute_similarity(get_embedding(updated), jd_vec)
    approved_count = sum(1 for c in changes if c.get("approved", False))
    return {
        "original_score" : original_score,
        "improved_score" : improved_score,
        "improvement"    : round(improved_score - original_score, 2),
        "approved_count" : approved_count,
        "updated_resume" : updated,
    }


@app.post("/generate-resume-pdf")
async def generate_resume_pdf(
    improved_resume_text: str = Form(...),
    job_description:      str = Form(...),
    candidate_name:       str = Form(default="Candidate"),
):
    """
    STEP 3 — Parse improved resume → build PDF with reportlab → serve download.
    Pure Python. No LibreOffice. No external software needed.
    """
    try:
        resume_json = extract_resume_as_json(improved_resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {e}")

    name     = resume_json.get("name", candidate_name).strip().replace(" ", "_")
    uid      = uuid.uuid4().hex[:8]
    pdf_path = os.path.join(TEMP_DIR, f"resume_{name}_{uid}.pdf")

    try:
        build_resume_pdf(resume_json, pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    filename = f"Improved_Resume_{name}_ResumeIQ.pdf"
    return FileResponse(
        path=pdf_path,
        filename=filename,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
