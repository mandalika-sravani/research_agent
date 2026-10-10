"""
Tools for the research agent.

Each tool has:
  - a Python function (the action)
  - a JSON schema (what the LLM sees)
  - a `critical` flag  -> critical tools need human approval (HITL)
  - a `timeout` in seconds
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
DOCS_DIR = BASE / "docs"
REPORTS_DIR = BASE / "reports"


# ---------------------------------------------------------------- tool 1
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Free web search via DuckDuckGo (no API key needed)."""
    try:
        from ddgs import DDGS                      # new package name
    except ImportError:
        from duckduckgo_search import DDGS         # old package name
    results = DDGS().text(query, max_results=max_results)
    return [{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
            for r in results]


# ---------------------------------------------------------------- tool 2
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the most relevant chunks from local files in ./docs.
    Simple keyword-overlap scoring so it runs anywhere.
    UPGRADE IDEA: swap this for your Module 6 embeddings + vector DB.
    """
    words = set(re.findall(r"\w+", query.lower()))
    chunks = []
    for f in DOCS_DIR.glob("*.*"):
        if f.suffix not in {".txt", ".md"}:
            continue
        for para in f.read_text(encoding="utf-8").split("\n\n"):
            para = para.strip()
            if not para:
                continue
            score = len(words & set(re.findall(r"\w+", para.lower())))
            if score:
                chunks.append({"source": f.name, "score": score, "text": para[:800]})
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:top_k] or [{"note": "No matching documents found."}]


# ---------------------------------------------------------------- tool 3 (CRITICAL)
def save_report(title: str, summary: str, key_findings: list, sources: list) -> dict:
    """Write the structured report to disk as Markdown + JSON."""
    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {"title": title, "summary": summary,
              "key_findings": key_findings, "sources": sources,
              "created_at": stamp}

    md = [f"# {title}", "", "## Summary", summary, "", "## Key findings"]
    md += [f"- {k}" for k in key_findings]
    md += ["", "## Sources"] + [f"- {s}" for s in sources]

    md_path = REPORTS_DIR / f"{stamp}-{slug}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    (REPORTS_DIR / f"{stamp}-{slug}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"saved_to": str(md_path)}


# ---------------------------------------------------------------- registry
TOOLS = {
    "web_search": {
        "fn": web_search, "critical": False, "timeout": 20,
        "schema": {
            "name": "web_search",
            "description": "Search the public web. Use for current facts, news, or anything not in local docs.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short search query, 2-8 words"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    "search_documents": {
        "fn": search_documents, "critical": False, "timeout": 10,
        "schema": {
            "name": "search_documents",
            "description": "Search the user's local documents folder. Try this before the web for internal topics.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    "save_report": {
        "fn": save_report, "critical": True, "timeout": 10,
        "schema": {
            "name": "save_report",
            "description": "Save the FINAL structured research report. Call exactly once, at the end.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string", "description": "3-5 sentence answer"},
                    "key_findings": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "sources": {"type": "array", "items": {"type": "string"},
                                "description": "URLs or doc filenames actually used"},
                },
                "required": ["title", "summary", "key_findings", "sources"],
            },
        },
    },
}

TYPE_MAP = {"string": str, "integer": int, "array": list, "object": dict, "boolean": bool}


def validate(name: str, args: dict) -> str | None:
    """Parameter validation. Returns an error message, or None if OK."""
    if name not in TOOLS:
        return f"Unknown tool '{name}'. Available: {list(TOOLS)}"
    schema = TOOLS[name]["schema"]["input_schema"]
    for req in schema.get("required", []):
        if req not in args:
            return f"Missing required parameter '{req}'"
    for key, val in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            return f"Unexpected parameter '{key}'"
        if not isinstance(val, TYPE_MAP[spec["type"]]):
            return f"Parameter '{key}' must be {spec['type']}"
        if spec["type"] == "integer" and not (spec.get("minimum", val) <= val <= spec.get("maximum", val)):
            return f"Parameter '{key}' out of range"
    return None


def execute(name: str, args: dict, retries: int = 2) -> dict:
    """Dispatch with validation, timeout and retry-with-backoff."""
    err = validate(name, args)
    if err:
        return {"ok": False, "error": f"Validation error: {err}"}

    tool = TOOLS[name]
    for attempt in range(retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(tool["fn"], **args).result(timeout=tool["timeout"])
            return {"ok": True, "result": result}
        except FuturesTimeout:
            last = f"Timed out after {tool['timeout']}s"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2 ** attempt)        # 1s, 2s ...
    return {"ok": False, "error": f"Tool failed after {retries + 1} attempts: {last}"}
