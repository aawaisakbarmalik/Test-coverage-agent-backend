Installation & Setup (Local)
1. Clone the repository

2. 
git clone https://github.com/your-username/Test-coverage-agent-backend.git
cd Test-coverage-agent-backend

 Create Virtual Environment
 
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # Linux/Mac


Install Requirements

pip install -r requirements.txt

Environment Variables

Create a .env file 

For Groq
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
LLM_MODEL=llama-3.1-8b-instant


Run the Backend Locally

Start FastAPI server:

uvicorn app.main:app --host 0.0.0.0 --port 8000


API will be served at:

http://127.0.0.1:8000


Docs UI:

http://127.0.0.1:8000/docs

 API Usage
1. Start an Analysis Task

Endpoint:

POST /api/ingest

Example (Git Repo)
curl -X POST "http://127.0.0.1:8000/api/ingest" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"python\",\"input_type\":\"git\",\"url_or_path\":\"https://github.com/user/project\",\"use_llm\":true}"


Example (Local files)
curl -X POST "http://127.0.0.1:8000/api/ingest" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"python\",\"input_type\":\"files\",\"url_or_path\":\"C:\\\\path\\\\to\\\\project\",\"use_llm\":true}"


Response contains a task_id.

2. Poll Task Status
GET /api/task/{task_id}


Example:

curl http://127.0.0.1:8000/api/task/<task_id>


The result includes:

Overall Coverage

Coverage By File

Generated tests

Suggested tests

LLM usage

Artifacts (coverage.json)
