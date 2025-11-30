from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from .worker import start_worker_sync, TASKS
import uuid
from datetime import datetime, timezone

app = FastAPI(title="Test Coverage Agent Supervisor", version="1.0.0")


class IngestRequest(BaseModel):
    language: Optional[str] = Field(default="python")
    input_type: Literal["git", "zip", "files"]
    url_or_path: str
    branch: Optional[str] = Field(default="main")


@app.post("/api/ingest")
async def ingest(req: IngestRequest, background: BackgroundTasks) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    task_assignment = {
        "message_id": str(uuid.uuid4()),
        "version": "1.0",
        "sender": "supervisor",
        "recipient": "worker",
        "type": "task_assignment",
        "related_message_id": None,
        "status": "queued",
        "task": {
            "task_id": task_id,
            "task_type": "ingest_and_analyze",
            "inputs": {
                "repo": {
                    "type": req.input_type,
                    "url_or_path": req.url_or_path,
                    "branch": req.branch,
                }
            },
            "parameters": {"language": req.language},
        },
        "results": None,
        "errors": None,
        "timestamp": now,
    }

    TASKS[task_id] = {"status": "queued", "assignment": task_assignment, "result": None}
    background.add_task(start_worker_sync, task_id, req.model_dump())
    return task_assignment

@app.get("/api/ingest")
async def ingest_status():
    return {"message": "Ingest endpoint is working", "status": "ready"}

@app.get("/api/task/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task_id,
        "status": task["status"],
        "assignment": task["assignment"],
        "result": task.get("result"),
    }
