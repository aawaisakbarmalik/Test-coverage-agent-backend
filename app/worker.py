# from __future__ import annotations

# import os
# import sys
# import re
# import json
# import uuid
# import shutil
# import zipfile
# import tempfile
# import subprocess
# from datetime import datetime, timezone
# from typing import Dict, Any, Tuple, List, Optional

# import ast
# import requests
# from git import Repo

# # =========================
# # In-memory task registry
# # =========================
# TASKS: Dict[str, Dict[str, Any]] = {}

# # =========================
# # LLM config (env-driven)
# # =========================
# LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()  # "openai" | "ollama" | "groq" | "gemini"
# LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or "gpt-4o-mini"

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()  # optional override

# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434/v1"

# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "").strip() or "https://api.groq.com/openai/v1"

# # Gemini v1 REST (NOT the old v1beta unless your model requires it)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "").strip() or "https://generativelanguage.googleapis.com"

# # Automatically use LLM if provider set
# USE_LLM_DEFAULT = bool(LLM_PROVIDER)

# # Debug logging to server console
# LLM_DEBUG = os.getenv("LLM_DEBUG", "0").strip() not in ("", "0", "false", "False")

# # Prompt limits (trim for big repos)
# MAX_SOURCES = int(os.getenv("PROMPT_MAX_SOURCES", "4"))   # how many worst-coverage files to include
# MAX_LINES_PER_SRC = int(os.getenv("PROMPT_MAX_LINES", "400"))
# MAX_EXISTING_TESTS = int(os.getenv("PROMPT_MAX_TESTS", "6"))

# # =========================
# # Helpers
# # =========================
# def _utcnow() -> str:
#     return datetime.now(timezone.utc).isoformat()

# def _safe_copytree(src: str, dst: str) -> None:
#     if not os.path.exists(src):
#         raise FileNotFoundError(f"source path not found: {src}")
#     shutil.copytree(src, dst, dirs_exist_ok=False)

# def _clone_git(url: str, branch: Optional[str], dest: str) -> None:
#     Repo.clone_from(url, dest, branch=branch or "main", depth=1)

# def _extract_zip(path: str, dest: str) -> None:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"zip path not found: {path}")
#     with zipfile.ZipFile(path, "r") as z:
#         z.extractall(dest)

# def _run_pytest_with_coverage(repo_dir: str) -> subprocess.CompletedProcess:
#     proc = subprocess.run(
#         [sys.executable, "-m", "coverage", "run", "-m", "pytest", "-q"],
#         cwd=repo_dir,
#         capture_output=True,
#         text=True,
#     )
#     subprocess.run(
#         [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
#         cwd=repo_dir,
#         capture_output=True,
#         text=True,
#     )
#     return proc

# def _parse_coverage_json(path_to_cov_json: str) -> Tuple[Optional[float], List[Dict[str, Any]], Dict[str, Any]]:
#     with open(path_to_cov_json, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     repo_root = os.path.abspath(os.path.dirname(path_to_cov_json))
#     totals = data.get("totals") or {}
#     files = data.get("files") or {}

#     per_file: List[Dict[str, Any]] = []
#     total_num = 0
#     total_cov = 0

#     for fpath, info in files.items():
#         norm = fpath.replace("\\", "/")
#         try:
#             rel = os.path.relpath(norm, start=repo_root).replace("\\", "/") if os.path.isabs(norm) else norm
#         except Exception:
#             rel = norm

#         summary = info.get("summary") or {}
#         num = summary.get("num_statements")
#         if num is None:
#             num = summary.get("num_lines", 0)
#         covered = summary.get("covered_lines", 0) or 0
#         cov = (covered / num) if num else 0.0

#         total_num += (num or 0)
#         total_cov += covered

#         per_file.append(
#             {
#                 "path": rel,
#                 "coverage": cov,
#                 "missing_lines": info.get("missing_lines", []) or [],
#                 "executed_lines": info.get("executed_lines", []) or [],
#             }
#         )

#     if totals and (totals.get("num_lines") or totals.get("num_statements")):
#         num_lines = totals.get("num_lines") or totals.get("num_statements") or 0
#         covered_lines = totals.get("covered_lines", 0) or 0
#         overall = (covered_lines / num_lines) if num_lines else None
#     else:
#         overall = (total_cov / total_num) if total_num else None

#     return overall, per_file, data

# # =========================
# # Code analysis helpers
# # =========================
# def _critical_score(fn_src: str, fn_name: str) -> int:
#     score = 0
#     score += len(re.findall(r"\bif\b|\bfor\b|\bwhile\b|\belif\b|\band\b|\bor\b", fn_src))
#     score += 3 * len(re.findall(r"\braise\b", fn_src))
#     if not fn_name.startswith("_"):
#         score += 1
#     if re.search(r"(pay|order|auth|login|total|discount|calc|validate|parse)", fn_name):
#         score += 2
#     return score

# def _py_functions_in_file(path: str) -> List[Dict[str, Any]]:
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             src = f.read()
#     except Exception:
#         return []
#     try:
#         tree = ast.parse(src, filename=path)
#     except Exception:
#         return []

#     lines = src.splitlines()
#     funcs: List[Dict[str, Any]] = []
#     for node in ast.walk(tree):
#         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#             start = getattr(node, "lineno", None)
#             end = getattr(node, "end_lineno", None)
#             snippet = ""
#             if start and end:
#                 snippet = "\n".join(lines[start - 1:end])
#             args = [a.arg for a in node.args.args]
#             funcs.append({
#                 "name": node.name,
#                 "start": start,
#                 "end": end,
#                 "args": args,
#                 "src": snippet,
#                 "criticality": _critical_score(snippet, node.name),
#             })
#     funcs.sort(key=lambda x: x.get("criticality", 0), reverse=True)
#     return funcs

# def _func_for_line(funcs: List[Dict[str, Any]], line: int) -> Optional[Dict[str, Any]]:
#     for f in funcs:
#         s, e = f.get("start"), f.get("end")
#         if s and e and s <= line <= e:
#             return f
#     return None

# def _read_existing_tests(repo_dir: str) -> List[Dict[str, str]]:
#     tests: List[Dict[str, str]] = []
#     for root, _, files in os.walk(repo_dir):
#         for fn in files:
#             if fn.endswith(".py") and (fn.startswith("test_") or os.path.basename(root) == "tests"):
#                 path = os.path.join(root, fn)
#                 try:
#                     with open(path, "r", encoding="utf-8") as f:
#                         tests.append({"path": os.path.relpath(path, repo_dir).replace("\\", "/"), "content": f.read()})
#                 except Exception:
#                     pass
#     return tests

# def _source_snippets_for_uncovered(repo_dir: str, per_file: List[Dict[str, Any]],
#                                    max_sources: int = MAX_SOURCES, max_lines: int = MAX_LINES_PER_SRC) -> str:
#     entries = [e for e in per_file if e["path"].endswith(".py") and not os.path.basename(e["path"]).startswith("test_")]
#     entries.sort(key=lambda e: e.get("coverage", 1.0))
#     blocks: List[str] = []

#     for e in entries[:max_sources]:
#         path = e["path"]
#         abspath = os.path.join(repo_dir, path.replace("/", os.sep))
#         if not os.path.exists(abspath):
#             continue
#         try:
#             code = open(abspath, "r", encoding="utf-8").read().splitlines()
#         except Exception:
#             continue

#         missing = set(e.get("missing_lines") or [])
#         annotated: List[str] = []
#         for i, line in enumerate(code, start=1):
#             prefix = "► " if i in missing else "  "
#             annotated.append(f"{prefix}{i:>4}: {line}")
#         if len(annotated) > max_lines:
#             annotated = annotated[:max_lines] + ["  ... [TRUNCATED] ..."]

#         blocks.append(
#             f"# File: {path} (coverage={e.get('coverage', 0.0)*100:.1f}%, missing={sorted(list(missing))[:30]})\n"
#             + "\n".join(annotated)
#         )

#     return "\n\n".join(blocks) if blocks else "# No source context"

# def _format_coverage_for_prompt(overall: Optional[float], per_file: List[Dict[str, Any]], raw_cov: Dict[str, Any]) -> str:
#     lines = []
#     lines.append(f"Overall coverage: {overall*100:.2f}%." if overall is not None else "Overall coverage: N/A.")
#     lines.append("Per-file coverage:")
#     for e in sorted(per_file, key=lambda x: x.get("coverage", 1.0)):
#         lines.append(f"- {e['path']}: {e.get('coverage', 0.0)*100:.1f}% (missing {len(e.get('missing_lines', []))} lines)")
#     compact = "\n".join(lines)
#     totals = raw_cov.get("totals", {})
#     raw_small = json.dumps({"totals": totals}, ensure_ascii=False)
#     return compact + "\n\nRaw coverage totals (for reference):\n" + raw_small

# # =========================
# # LLM prompt + parsing
# # =========================
# _CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

# def _extract_code_block(text: str) -> Optional[str]:
#     if not text:
#         return None
#     m = _CODE_BLOCK_RE.search(text)
#     if m:
#         return m.group(1).strip()
#     # fallback: if no fencing, assume raw
#     return text.strip() if "def " in text or "import " in text else None

# def _build_llm_prompt(
#     repo_dir: str,
#     overall: Optional[float],
#     per_file: List[Dict[str, Any]],
#     raw_cov: Dict[str, Any],
#     existing_tests: List[Dict[str, str]],
# ) -> str:
#     coverage_section = _format_coverage_for_prompt(overall, per_file, raw_cov)
#     snippets = _source_snippets_for_uncovered(repo_dir, per_file)
#     tests_parts = []
#     for t in existing_tests[:MAX_EXISTING_TESTS]:
#         tests_parts.append(f"# Existing test: {t['path']}\n{t['content']}")
#     tests_blob = "\n\n".join(tests_parts) if tests_parts else "# No existing tests found"

#     rules = (
#         "You are a senior Python engineer.\n"
#         "Inputs you receive:\n"
#         "1) Coverage summary\n"
#         "2) Source snippets with uncovered lines marked by '►'\n"
#         "3) Existing tests (if any)\n\n"
#         "TASK: Generate ONE pytest module that meaningfully increases coverage, focusing on the uncovered logic.\n"
#         "HARD REQUIREMENTS:\n"
#         "- Output ONLY a single fenced Python code block, nothing else.\n"
#         "- The module must be runnable as-is under pytest.\n"
#         "- Import modules by filename (e.g., from calc import add).\n"
#         "- No network/filesystem; keep tests deterministic.\n"
#         "- Prefer boundary cases and error branches.\n"
#         "- Use pytest.raises for exceptions.\n"
#     )

#     user = (
#         f"{rules}\n\n"
#         f"== Coverage ==\n{coverage_section}\n\n"
#         f"== Source (uncovered annotated) ==\n{snippets}\n\n"
#         f"== Existing tests ==\n{tests_blob}\n\n"
#         "Return only the code block like:\n"
#         "```python\n# your tests here\n```\n"
#     )
#     return user

# # =========================
# # LLM backends
# # =========================
# def _debug_log(title: str, payload: Any):
#     if LLM_DEBUG:
#         try:
#             print(f"[LLM_DEBUG] {title}: {json.dumps(payload, ensure_ascii=False)[:4000]}", flush=True)
#         except Exception:
#             print(f"[LLM_DEBUG] {title}: (non-serializable)", flush=True)

# def _call_llm(prompt: str, provider: str, model: str, temperature: float = 0.2) -> Optional[str]:
#     provider = (provider or "").lower().strip()
#     try:
#         if provider == "openai":
#             base = OPENAI_BASE_URL or "https://api.openai.com/v1"
#             url = f"{base}/chat/completions"
#             headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
#             body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
#             _debug_log("OPENAI_REQ", {"url": url, "model": model})
#             resp = requests.post(url, headers=headers, json=body, timeout=180)
#             _debug_log("OPENAI_STATUS", {"status": resp.status_code})
#             if resp.status_code >= 400:
#                 _debug_log("OPENAI_ERR", resp.text)
#                 return None
#             data = resp.json()
#             return data["choices"][0]["message"]["content"]

#         elif provider == "ollama":
#             base = OLLAMA_BASE_URL  # e.g., http://localhost:11434/v1
#             url = f"{base}/chat/completions"
#             body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
#             _debug_log("OLLAMA_REQ", {"url": url, "model": model})
#             resp = requests.post(url, json=body, timeout=180)
#             _debug_log("OLLAMA_STATUS", {"status": resp.status_code})
#             if resp.status_code >= 400:
#                 _debug_log("OLLAMA_ERR", resp.text)
#                 return None
#             data = resp.json()
#             return data["choices"][0]["message"]["content"]

#         elif provider == "groq":
#             base = GROQ_BASE_URL  # https://api.groq.com/openai/v1
#             url = f"{base}/chat/completions"
#             headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
#             body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
#             _debug_log("GROQ_REQ", {"url": url, "model": model})
#             resp = requests.post(url, headers=headers, json=body, timeout=180)
#             _debug_log("GROQ_STATUS", {"status": resp.status_code})
#             if resp.status_code >= 400:
#                 _debug_log("GROQ_ERR", resp.text)
#                 return None
#             data = resp.json()
#             return data["choices"][0]["message"]["content"]

#         elif provider == "gemini":
#             # Gemini v1 REST: POST /v1/models/{model}:generateContent?key=API_KEY
#             # model must be e.g. "gemini-2.5-flash" or "gemini-2.5-pro" (see GET /v1/models)
#             api_key = GEMINI_API_KEY
#             if not api_key:
#                 _debug_log("GEMINI_ERR", "No GEMINI_API_KEY set")
#                 return None
#             base = GEMINI_BASE_URL.rstrip("/")
#             url = f"{base}/v1/models/{model}:generateContent?key={api_key}"
#             body = {
#                 "contents": [
#                     {"role": "user", "parts": [{"text": prompt}]}
#                 ],
#                 "generationConfig": {
#                     "temperature": temperature
#                 }
#             }
#             _debug_log("GEMINI_REQ", {"url": url, "model": model})
#             resp = requests.post(url, json=body, timeout=180)
#             _debug_log("GEMINI_STATUS", {"status": resp.status_code})
#             if resp.status_code >= 400:
#                 _debug_log("GEMINI_ERR", resp.text)
#                 return None
#             data = resp.json()
#             # Concatenate all text parts
#             text = ""
#             for cand in (data.get("candidates") or []):
#                 parts = cand.get("content", {}).get("parts", [])
#                 for p in parts:
#                     if "text" in p:
#                         text += p["text"]
#             return text or None

#         else:
#             _debug_log("LLM_PROVIDER_ERR", f"Unsupported provider: {provider}")
#             return None

#     except Exception as e:
#         _debug_log("LLM_EXCEPTION", str(e))
#         return None

# def _write_generated_file(repo_dir: str, suggested_path: str, content: str) -> str:
#     out_path = os.path.join(repo_dir, suggested_path.replace("/", os.sep))
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(content)
#     return out_path

# # =========================
# # Task response wrapper
# # =========================
# def _build_task_response(
#     assignment: Dict[str, Any],
#     *,
#     success: bool,
#     results: Optional[Dict[str, Any]] = None,
#     errors: Optional[str] = None,
# ) -> Dict[str, Any]:
#     return {
#         "message_id": str(uuid.uuid4()),
#         "version": assignment.get("version", "1.0"),
#         "sender": "worker",
#         "recipient": "supervisor",
#         "type": "task_response",
#         "related_message_id": assignment.get("message_id"),
#         "status": "success" if success else "failed",
#         "task": assignment.get("task"),
#         "results": results,
#         "errors": errors,
#         "timestamp": _utcnow(),
#     }

# # =========================
# # Worker entry
# # =========================
# def start_worker_sync(task_id: str, payload: Dict[str, Any]) -> None:
#     record = TASKS.get(task_id)
#     if not record:
#         return

#     record["status"] = "running"
#     assignment = record["assignment"]

#     tmp = tempfile.mkdtemp(prefix="tca_")
#     repo_dir = os.path.join(tmp, "repo")

#     try:
#         input_type = payload.get("input_type")
#         url_or_path = payload.get("url_or_path")
#         branch = payload.get("branch")
#         use_llm = bool(payload.get("use_llm", USE_LLM_DEFAULT))
#         llm_model = str(payload.get("llm_model") or LLM_MODEL)
#         write_generated = bool(payload.get("write_generated", False))
#         rerun_after_write = bool(payload.get("rerun_after_write", True))

#         # fetch repo
#         if input_type == "git":
#             _clone_git(url_or_path, branch, repo_dir)
#         elif input_type == "zip":
#             os.makedirs(repo_dir, exist_ok=True)
#             _extract_zip(url_or_path, repo_dir)
#         elif input_type == "files":
#             _safe_copytree(url_or_path, repo_dir)
#         else:
#             raise RuntimeError(f"unsupported input_type: {input_type}")

#         # first coverage run (baseline)
#         proc = _run_pytest_with_coverage(repo_dir)
#         cov_json = os.path.join(repo_dir, "coverage.json")

#         if not os.path.exists(cov_json):
#             res = {
#                 "summary": {
#                     "note": "coverage.json not produced (no tests or failures).",
#                     "pytest_returncode": proc.returncode,
#                     "pytest_stdout": proc.stdout,
#                     "pytest_stderr": proc.stderr,
#                 }
#             }
#             record["status"] = "finished"
#             record["result"] = _build_task_response(
#                 assignment, success=False, results=res, errors="coverage json missing"
#             )
#             return

#         overall, per_file, raw_cov = _parse_coverage_json(cov_json)
#         existing_tests = _read_existing_tests(repo_dir)

#         # ========== LLM: propose tests ==========
#         llm_piece: Dict[str, Any] = {"used": False}
#         generated_code: Optional[str] = None
#         suggested_path = "tests/test_from_agent.py"

#         if use_llm and LLM_PROVIDER in ("openai", "ollama", "groq", "gemini"):
#             prompt = _build_llm_prompt(repo_dir, overall, per_file, raw_cov, existing_tests)
#             raw = _call_llm(prompt, LLM_PROVIDER, llm_model, temperature=0.2)
#             code = _extract_code_block(raw)

#             # retry with higher temperature if empty
#             if not code:
#                 raw2 = _call_llm(
#                     prompt + "\n\nIMPORTANT: Output ONLY a single fenced Python code block now.",
#                     LLM_PROVIDER, llm_model, temperature=0.6
#                 )
#                 code = _extract_code_block(raw2)

#             if code:
#                 generated_code = code
#                 llm_piece = {"provider": LLM_PROVIDER, "model": llm_model, "used": True}
#             else:
#                 llm_piece = {"provider": LLM_PROVIDER, "model": llm_model, "used": False, "note": "no code returned"}

#         # write file if requested
#         written_path = None
#         if generated_code and write_generated:
#             written_path = _write_generated_file(repo_dir, suggested_path, generated_code)
#             if rerun_after_write:
#                 _run_pytest_with_coverage(repo_dir)
#                 overall, per_file, raw_cov = _parse_coverage_json(os.path.join(repo_dir, "coverage.json"))

#         # suggestions list (LLM-centric or fallback)
#         suggestions: List[Dict[str, Any]] = []
#         if generated_code:
#             suggestions.append({
#                 "id": str(uuid.uuid4()),
#                 "target": "generated_by_llm",
#                 "description": "AI-generated tests to improve coverage based on uncovered lines and existing tests.",
#                 "test_template": generated_code,
#                 "priority": "ai",
#             })
#         else:
#             for e in sorted(per_file, key=lambda x: x.get("coverage", 1.0))[:3]:
#                 suggestions.append({
#                     "id": str(uuid.uuid4()),
#                     "target": f"{e['path']}",
#                     "description": "Suggest writing a focused test around uncovered lines.",
#                     "test_template": "# LLM returned no code; verify provider/env; reduce prompt size; retry.",
#                     "priority": "high",
#                 })

#         results = {
#             "summary": {
#                 "overall_coverage": overall,
#                 "total_files": len(per_file),
#                 "critical_suggestions": sum(1 for s in suggestions if s["priority"] in ("ai", "critical", "high")),
#                 "message": "Test Coverage Agent – LLM proposes new tests based on coverage + existing tests.",
#             },
#             "coverage_by_file": per_file,
#             "suggested_tests": suggestions,
#             "generated_tests_file": {
#                 "suggested_path": suggested_path,
#                 "content": generated_code or "",
#                 "written_to": written_path,
#             },
#             "llm": llm_piece,
#             "artifacts": [{"type": "coverage_report", "format": "json", "path": os.path.join(repo_dir, "coverage.json")}],
#         }

#         record["status"] = "finished"
#         record["result"] = _build_task_response(assignment, success=True, results=results)

#     except Exception as e:
#         record["status"] = "failed"
#         record["result"] = _build_task_response(
#             assignment, success=False, results=None, errors=str(e)
#         )
#     finally:
#         # keep tmp dir for inspection
#         pass











from __future__ import annotations

import os
import sys
import re
import json
import uuid
import shutil
import zipfile
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional

import ast
import requests
from git import Repo

# =========================
# In-memory task registry
# =========================
TASKS: Dict[str, Dict[str, Any]] = {}

# =========================
# LLM config (env-driven)
# =========================
# Supported providers: "openai" | "ollama" | "groq" | "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or "gpt-4o-mini"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"

# Ollama (OpenAI-compatible /chat/completions endpoint via /v1)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434/v1"

# Groq (OpenAI-compatible)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "").strip() or "https://api.groq.com/openai/v1"

# Gemini v1 REST
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "").strip() or "https://generativelanguage.googleapis.com"

# Automatically use LLM if provider set
USE_LLM_DEFAULT = bool(LLM_PROVIDER)

# Debug logging to server console (set LLM_DEBUG=1)
LLM_DEBUG = os.getenv("LLM_DEBUG", "0").strip() not in ("", "0", "false", "False")

# Prompt budget knobs (shorter = safer for big repos)
MAX_SOURCES = int(os.getenv("PROMPT_MAX_SOURCES", "2"))     # worst-coverage files to include
MAX_LINES_PER_SRC = int(os.getenv("PROMPT_MAX_LINES", "300"))
MAX_EXISTING_TESTS = int(os.getenv("PROMPT_MAX_TESTS", "3"))

# Optional: limit which files are considered (simple substring match)
INCLUDE_PATH_SUBSTR = os.getenv("INCLUDE_PATH_SUBSTR", "").strip()

# =========================
# Helpers
# =========================
def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_copytree(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise FileNotFoundError(f"source path not found: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=False)

def _clone_git(url: str, branch: Optional[str], dest: str) -> None:
    Repo.clone_from(url, dest, branch=branch or "main", depth=1)

def _extract_zip(path: str, dest: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"zip path not found: {path}")
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(dest)

def _run_pytest_with_coverage(repo_dir: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "run", "-m", "pytest", "-q"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    return proc

def _parse_coverage_json(path_to_cov_json: str) -> Tuple[Optional[float], List[Dict[str, Any]], Dict[str, Any]]:
    with open(path_to_cov_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    repo_root = os.path.abspath(os.path.dirname(path_to_cov_json))
    totals = data.get("totals") or {}
    files = data.get("files") or {}

    per_file: List[Dict[str, Any]] = []
    total_num = 0
    total_cov = 0

    for fpath, info in files.items():
        norm = fpath.replace("\\", "/")
        try:
            rel = os.path.relpath(norm, start=repo_root).replace("\\", "/") if os.path.isabs(norm) else norm
        except Exception:
            rel = norm

        summary = info.get("summary") or {}
        num = summary.get("num_statements")
        if num is None:
            num = summary.get("num_lines", 0)
        covered = summary.get("covered_lines", 0) or 0
        cov = (covered / num) if num else 0.0

        total_num += (num or 0)
        total_cov += covered

        per_file.append(
            {
                "path": rel,
                "coverage": cov,
                "missing_lines": info.get("missing_lines", []) or [],
                "executed_lines": info.get("executed_lines", []) or [],
            }
        )

    if totals and (totals.get("num_lines") or totals.get("num_statements")):
        num_lines = totals.get("num_lines") or totals.get("num_statements") or 0
        covered_lines = totals.get("covered_lines", 0) or 0
        overall = (covered_lines / num_lines) if num_lines else None
    else:
        overall = (total_cov / total_num) if total_num else None

    # optional: filter per_file to reduce prompt size
    if INCLUDE_PATH_SUBSTR:
        per_file = [e for e in per_file if INCLUDE_PATH_SUBSTR in e["path"]]

    return overall, per_file, data

# =========================
# Code analysis helpers
# =========================
def _critical_score(fn_src: str, fn_name: str) -> int:
    score = 0
    score += len(re.findall(r"\bif\b|\bfor\b|\bwhile\b|\belif\b|\band\b|\bor\b", fn_src))
    score += 3 * len(re.findall(r"\braise\b", fn_src))
    if not fn_name.startswith("_"):
        score += 1
    if re.search(r"(pay|order|auth|login|total|discount|calc|validate|parse)", fn_name):
        score += 2
    return score

def _py_functions_in_file(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return []
    try:
        tree = ast.parse(src, filename=path)
    except Exception:
        return []

    lines = src.splitlines()
    funcs: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            snippet = ""
            if start and end:
                snippet = "\n".join(lines[start - 1:end])
            args = [a.arg for a in node.args.args]
            funcs.append({
                "name": node.name,
                "start": start,
                "end": end,
                "args": args,
                "src": snippet,
                "criticality": _critical_score(snippet, node.name),
            })
    funcs.sort(key=lambda x: x.get("criticality", 0), reverse=True)
    return funcs

def _func_for_line(funcs: List[Dict[str, Any]], line: int) -> Optional[Dict[str, Any]]:
    for f in funcs:
        s, e = f.get("start"), f.get("end")
        if s and e and s <= line <= e:
            return f
    return None

def _read_existing_tests(repo_dir: str) -> List[Dict[str, str]]:
    tests: List[Dict[str, str]] = []
    for root, _, files in os.walk(repo_dir):
        for fn in files:
            if fn.endswith(".py") and (fn.startswith("test_") or os.path.basename(root) == "tests"):
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tests.append({"path": os.path.relpath(path, repo_dir).replace("\\", "/"), "content": f.read()})
                except Exception:
                    pass
    return tests

def _source_snippets_for_uncovered(repo_dir: str, per_file: List[Dict[str, Any]],
                                   max_sources: int = MAX_SOURCES, max_lines: int = MAX_LINES_PER_SRC) -> str:
    entries = [e for e in per_file if e["path"].endswith(".py") and not os.path.basename(e["path"]).startswith("test_")]
    # worst coverage first
    entries.sort(key=lambda e: e.get("coverage", 1.0))
    blocks: List[str] = []

    for e in entries[:max_sources]:
        path = e["path"]
        abspath = os.path.join(repo_dir, path.replace("/", os.sep))
        if not os.path.exists(abspath):
            continue
        try:
            code = open(abspath, "r", encoding="utf-8").read().splitlines()
        except Exception:
            continue

        missing = set(e.get("missing_lines") or [])
        annotated: List[str] = []
        # keep lines short to avoid token blow-up
        for i, line in enumerate(code, start=1):
            line = line[:200]  # hard cap per line to avoid massive prompts
            prefix = "► " if i in missing else "  "
            annotated.append(f"{prefix}{i:>4}: {line}")
        if len(annotated) > max_lines:
            annotated = annotated[:max_lines] + ["  ... [TRUNCATED] ..."]

        blocks.append(
            f"# File: {path} (coverage={e.get('coverage', 0.0)*100:.1f}%, missing={sorted(list(missing))[:30]})\n"
            + "\n".join(annotated)
        )

    return "\n\n".join(blocks) if blocks else "# No source context"

def _format_coverage_for_prompt(overall: Optional[float], per_file: List[Dict[str, Any]], raw_cov: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Overall coverage: {overall*100:.2f}%." if overall is not None else "Overall coverage: N/A.")
    lines.append("Per-file coverage (lowest first):")
    for e in sorted(per_file, key=lambda x: x.get("coverage", 1.0)):
        lines.append(f"- {e['path']}: {e.get('coverage', 0.0)*100:.1f}% (missing {len(e.get('missing_lines', []))} lines)")
    compact = "\n".join(lines)
    totals = raw_cov.get("totals", {})
    raw_small = json.dumps({"totals": totals}, ensure_ascii=False)
    return compact + "\n\nRaw coverage totals (for reference):\n" + raw_small

# =========================
# LLM prompt + parsing
# =========================
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

def _extract_code_block(text: str) -> Optional[str]:
    if not text:
        return None
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: if no fencing, assume raw python-like content
    t = text.strip()
    if "def " in t or "import " in t or "pytest" in t:
        return t
    return None

def _build_llm_prompt(
    repo_dir: str,
    overall: Optional[float],
    per_file: List[Dict[str, Any]],
    raw_cov: Dict[str, Any],
    existing_tests: List[Dict[str, str]],
) -> str:
    coverage_section = _format_coverage_for_prompt(overall, per_file, raw_cov)
    snippets = _source_snippets_for_uncovered(repo_dir, per_file)

    tests_parts = []
    for t in existing_tests[:MAX_EXISTING_TESTS]:
        # keep existing tests compact
        body = t["content"]
        if len(body) > 2000:
            body = body[:2000] + "\n# ... [TRUNCATED] ..."
        tests_parts.append(f"# Existing test: {t['path']}\n{body}")
    tests_blob = "\n\n".join(tests_parts) if tests_parts else "# No existing tests found"

    rules = (
        "You are a senior Python engineer.\n"
        "Inputs:\n"
        "1) Coverage summary\n"
        "2) Source snippets with uncovered lines marked by '►'\n"
        "3) Existing tests (if any)\n\n"
        "TASK: Produce ONE pytest module that increases coverage, focusing on uncovered logic and branches.\n"
        "HARD RULES:\n"
        "- Output ONLY a single fenced Python code block. No prose.\n"
        "- The module must be runnable with pytest (no network/filesystem).\n"
        "- Import modules by filename (e.g., from calc import add).\n"
        "- Prefer boundary cases and error branches; use pytest.raises for exceptions.\n"
    )

    user = (
        f"{rules}\n\n"
        f"== Coverage ==\n{coverage_section}\n\n"
        f"== Source (uncovered annotated) ==\n{snippets}\n\n"
        f"== Existing tests ==\n{tests_blob}\n\n"
        "Return only the code block like:\n"
        "```python\n# tests/test_from_agent.py\n...\n```\n"
    )
    return user

# =========================
# LLM backends
# =========================
def _debug_log(title: str, payload: Any):
    if LLM_DEBUG:
        try:
            print(f"[LLM_DEBUG] {title}: {json.dumps(payload, ensure_ascii=False)[:4000]}", flush=True)
        except Exception:
            print(f"[LLM_DEBUG] {title}: (non-serializable)", flush=True)

def _call_llm(prompt: str, provider: str, model: str, temperature: float = 0.2) -> Optional[str]:
    provider = (provider or "").lower().strip()
    try:
        if provider == "openai":
            url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
            _debug_log("OPENAI_REQ", {"url": url, "model": model})
            resp = requests.post(url, headers=headers, json=body, timeout=180)
            _debug_log("OPENAI_STATUS", {"status": resp.status_code})
            if resp.status_code >= 400:
                _debug_log("OPENAI_ERR", resp.text)
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        elif provider == "ollama":
            url = f"{OLLAMA_BASE_URL.rstrip('/')}/chat/completions"
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
            _debug_log("OLLAMA_REQ", {"url": url, "model": model})
            resp = requests.post(url, json=body, timeout=180)
            _debug_log("OLLAMA_STATUS", {"status": resp.status_code})
            if resp.status_code >= 400:
                _debug_log("OLLAMA_ERR", resp.text)
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        elif provider == "groq":
            # Recommend model: llama-3.1-8b-instant (the older llama3-8b-8192 is decommissioned)
            url = f"{GROQ_BASE_URL.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
            _debug_log("GROQ_REQ", {"url": url, "model": model})
            resp = requests.post(url, headers=headers, json=body, timeout=180)
            _debug_log("GROQ_STATUS", {"status": resp.status_code})
            if resp.status_code >= 400:
                _debug_log("GROQ_ERR", resp.text)
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        elif provider == "gemini":
            # Gemini v1 REST: POST /v1/models/{model}:generateContent?key=API_KEY
            # Example valid models (free tier may be rate-limited): gemini-2.5-flash
            if not GEMINI_API_KEY:
                _debug_log("GEMINI_ERR", "No GEMINI_API_KEY set")
                return None
            base = GEMINI_BASE_URL.rstrip("/")
            url = f"{base}/v1/models/{model}:generateContent?key={GEMINI_API_KEY}"
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature}
            }
            _debug_log("GEMINI_REQ", {"url": url, "model": model})
            resp = requests.post(url, json=body, timeout=180)
            _debug_log("GEMINI_STATUS", {"status": resp.status_code})
            if resp.status_code >= 400:
                _debug_log("GEMINI_ERR", resp.text)
                return None
            data = resp.json()
            text = ""
            for cand in (data.get("candidates") or []):
                parts = cand.get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        text += p["text"]
            return text or None

        else:
            _debug_log("LLM_PROVIDER_ERR", f"Unsupported provider: {provider}")
            return None

    except Exception as e:
        _debug_log("LLM_EXCEPTION", str(e))
        return None

def _write_generated_file(repo_dir: str, suggested_path: str, content: str) -> str:
    out_path = os.path.join(repo_dir, suggested_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path

# =========================
# Task response wrapper
# =========================
def _build_task_response(
    assignment: Dict[str, Any],
    *,
    success: bool,
    results: Optional[Dict[str, Any]] = None,
    errors: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "message_id": str(uuid.uuid4()),
        "version": assignment.get("version", "1.0"),
        "sender": "worker",
        "recipient": "supervisor",
        "type": "task_response",
        "related_message_id": assignment.get("message_id"),
        "status": "success" if success else "failed",
        "task": assignment.get("task"),
        "results": results,
        "errors": errors,
        "timestamp": _utcnow(),
    }

# =========================
# Worker entry
# =========================
def start_worker_sync(task_id: str, payload: Dict[str, Any]) -> None:
    record = TASKS.get(task_id)
    if not record:
        return

    record["status"] = "running"
    assignment = record["assignment"]

    tmp = tempfile.mkdtemp(prefix="tca_")
    repo_dir = os.path.join(tmp, "repo")

    try:
        input_type = payload.get("input_type")
        url_or_path = payload.get("url_or_path")
        branch = payload.get("branch")
        use_llm = bool(payload.get("use_llm", USE_LLM_DEFAULT))
        llm_model = str(payload.get("llm_model") or LLM_MODEL)
        write_generated = bool(payload.get("write_generated", False))
        rerun_after_write = bool(payload.get("rerun_after_write", True))

        # fetch repo
        if input_type == "git":
            _clone_git(url_or_path, branch, repo_dir)
        elif input_type == "zip":
            os.makedirs(repo_dir, exist_ok=True)
            _extract_zip(url_or_path, repo_dir)
        elif input_type == "files":
            _safe_copytree(url_or_path, repo_dir)
        else:
            raise RuntimeError(f"unsupported input_type: {input_type}")

        # first coverage run (baseline)
        proc = _run_pytest_with_coverage(repo_dir)
        cov_json = os.path.join(repo_dir, "coverage.json")

        if not os.path.exists(cov_json):
            res = {
                "summary": {
                    "note": "coverage.json not produced (no tests or failures).",
                    "pytest_returncode": proc.returncode,
                    "pytest_stdout": proc.stdout,
                    "pytest_stderr": proc.stderr,
                }
            }
            record["status"] = "finished"
            record["result"] = _build_task_response(
                assignment, success=False, results=res, errors="coverage json missing"
            )
            return

        overall, per_file, raw_cov = _parse_coverage_json(cov_json)
        existing_tests = _read_existing_tests(repo_dir)

        # ========== LLM: propose tests ==========
        llm_piece: Dict[str, Any] = {"used": False}
        generated_code: Optional[str] = None
        suggested_path = "tests/test_from_agent.py"

        if use_llm and LLM_PROVIDER in ("openai", "ollama", "groq", "gemini"):
            prompt = _build_llm_prompt(repo_dir, overall, per_file, raw_cov, existing_tests)
            raw = _call_llm(prompt, LLM_PROVIDER, llm_model, temperature=0.2)
            code = _extract_code_block(raw)

            # retry with higher temperature if empty
            if not code:
                raw2 = _call_llm(
                    prompt + "\n\nIMPORTANT: Output ONLY a single fenced Python code block now.",
                    LLM_PROVIDER, llm_model, temperature=0.6
                )
                code = _extract_code_block(raw2)

            if code:
                generated_code = code
                llm_piece = {"provider": LLM_PROVIDER, "model": llm_model, "used": True}
            else:
                llm_piece = {"provider": LLM_PROVIDER, "model": llm_model, "used": False, "note": "no code returned"}

        # write file if requested
        written_path = None
        if generated_code and write_generated:
            written_path = _write_generated_file(repo_dir, suggested_path, generated_code)
            if rerun_after_write:
                _run_pytest_with_coverage(repo_dir)
                overall, per_file, raw_cov = _parse_coverage_json(os.path.join(repo_dir, "coverage.json"))

        # suggestions list
        suggestions: List[Dict[str, Any]] = []
        if generated_code:
            suggestions.append({
                "id": str(uuid.uuid4()),
                "target": "generated_by_llm",
                "description": "AI-generated tests to improve coverage based on uncovered lines and existing tests.",
                "test_template": generated_code,
                "priority": "ai",
            })
        else:
            # fallback: small heuristic nudge
            for e in sorted(per_file, key=lambda x: x.get("coverage", 1.0))[:3]:
                suggestions.append({
                    "id": str(uuid.uuid4()),
                    "target": f"{e['path']}",
                    "description": "Suggest writing a focused test around uncovered lines.",
                    "test_template": "# LLM returned no code; verify provider/env; reduce prompt size; retry.",
                    "priority": "high",
                })

        results = {
            "summary": {
                "overall_coverage": overall,
                "total_files": len(per_file),
                "critical_suggestions": sum(1 for s in suggestions if s["priority"] in ("ai", "critical", "high")),
                "message": "Test Coverage Agent – LLM proposes new tests based on coverage + existing tests.",
            },
            "coverage_by_file": per_file,
            "suggested_tests": suggestions,
            "generated_tests_file": {
                "suggested_path": suggested_path,
                "content": generated_code or "",
                "written_to": written_path,
            },
            "llm": llm_piece,
            "artifacts": [{"type": "coverage_report", "format": "json", "path": os.path.join(repo_dir, "coverage.json")}],
        }

        record["status"] = "finished"
        record["result"] = _build_task_response(assignment, success=True, results=results)

    except Exception as e:
        record["status"] = "failed"
        record["result"] = _build_task_response(
            assignment, success=False, results=None, errors=str(e)
        )
    finally:
        # keep tmp dir for inspection
        pass
