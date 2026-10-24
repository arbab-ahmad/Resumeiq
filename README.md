# ResumeIQ — AI Resume Analyzer

An AI-powered resume scoring and improvement tool built with FastAPI, Sentence Transformers, FAISS, and Groq AI.

## What It Does

- Scores your resume against a job description (0–100%)
- Suggests improvements using AI and expert resume tips
- Classifies every bullet point as Strong, Moderate, or Weak
- Lets you accept or reject each suggested rewrite
- Shows before and after score after improvements
- Exports a professional PDF of your improved resume
- Recruiter mode ranks multiple candidates by match score

## Setup

**1. Clone the repo and create a virtual environment**
```bash
cd resumeiq
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get a free Groq API key at [console.groq.com](https://console.groq.com) and set it**
```bash
export GROQ_API_KEY="gsk_your_key_here"    # Mac / Linux
set GROQ_API_KEY=gsk_your_key_here         # Windows
```

**4. Start the server**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**5. Open your browser at `http://localhost:8000`**

> First run downloads the embedding model (~90MB). This only happens once.

## Usage

**Applicant Mode**
1. Upload your resume (PDF or DOCX)
2. Paste the job description
3. Click **Analyze Match** to see your score and suggestions
4. Click **Improve My Resume** to review bullet rewrites
5. Accept or reject each change individually
6. Click **Apply Accepted Changes** to see your new score
7. Download the improved resume as a PDF

**Recruiter Mode**
1. Switch to Recruiter Mode
2. Upload 2–5 resumes
3. Paste the job description
4. Click **Rank Candidates**

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI, Python 3.10+ |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| AI Model | Groq (llama-3.3-70b-versatile) |
| PDF Parsing | PyMuPDF, python-docx |
| PDF Export | reportlab |
| Frontend | HTML, CSS, JavaScript |

## Demo Files

Ready-made test files are in the `demo files/` folder:

| File | Expected Score |
|---|---|
| Resume_HIGH_Ayesha_Khan.pdf | ~81% |
| Resume_AVG_Ravi_Sharma.pdf | ~48% |
| Resume_POOR_Suresh_Mehta.pdf | ~9% |

Use `Job_Description_TechVision.pdf` as the test job description.

## Notes

- Requires an internet connection for Groq API calls
- Only works with text-based PDFs (not scanned images)
- Resume data is never saved — each session is stateless
- Never hardcode your API key in the source code
