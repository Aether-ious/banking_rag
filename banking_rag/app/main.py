import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .rag_engine import process_banking_doc, get_compliance_answer

app = FastAPI(title="Banking Policy Compliance API")

class QueryRequest(BaseModel):
    question: str

@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    """Upload a Credit/Loan Policy PDF."""
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        count = process_banking_doc(tmp_path)
        return {"status": "success", "clauses_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@app.post("/check-compliance")
async def check_compliance(request: QueryRequest):
    """Ask a question: 'Is a 650 score eligible for a Home Loan?'"""
    try:
        return get_compliance_answer(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))