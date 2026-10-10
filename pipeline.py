"""
Deterministic research PIPELINE: the same job as agent.py, but with a fixed path.

    search_documents -> web_search -> ONE LLM call writes the report -> human approval -> save

The code decides every step. The LLM only writes the report; it never chooses tools,
never loops and never decides when to stop. Compare it with agent.py, where the
model chooses which tool to call next and when it is done.

Run:  python pipeline.py "your research question"
"""
import json
import os
import sys
import time

from anthropic import Anthropic
from dotenv import load_dotenv

from tools import TOOLS, execute

load_dotenv()
MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")


def ask_human(args):
    """[HITL] Same approval gate as the agent, without the feedback loop (a pipeline doesn't loop)."""
    print("\n" + "=" * 60)
    print("APPROVAL NEEDED: pipeline wants to run 'save_report'")
    print(json.dumps(args, indent=2)[:2000])
    print("=" * 60)
    return input("[a]pprove / [r]eject: ").strip().lower().startswith("a")


def run_pipeline(question, client=None):
    client = client or Anthropic()
    start, tokens = time.time(), 0

    # Step 1: always search local docs with the question as-is
    print("1/4  search_documents")
    docs = execute("search_documents", {"query": question, "top_k": 5})

    # Step 2: always search the web with the question as-is
    print("2/4  web_search")
    web = execute("web_search", {"query": question, "max_results": 6})

    # Step 3: ONE LLM call. The only tool offered is save_report, so its JSON schema
    # shapes the output. (Forcing it with tool_choice isn't supported on every model.)
    print("3/4  write report (1 LLM call)")
    resp = client.messages.create(
        model=MODEL, max_tokens=3000,
        tools=[TOOLS["save_report"]["schema"]],
        messages=[{"role": "user", "content":
            f"Question: {question}\n\nLocal documents:\n{json.dumps(docs)[:6000]}\n\n"
            f"Web results:\n{json.dumps(web)[:6000]}\n\n"
            "Write a research report that answers the question using ONLY this evidence. "
            "Cite only sources that appear above, and label your own inferences as such. "
            "Return the report by calling the save_report tool. Do not reply in plain text."}],
    )
    tokens += resp.usage.input_tokens + resp.usage.output_tokens
    report = next((b.input for b in resp.content if b.type == "tool_use"), None)
    if report is None:   # a pipeline has no loop to recover in, so fail clearly
        print("Model did not return a structured report. Pipeline stopped.")
        return "error"

    # Step 4: human approval, then save
    print("4/4  human approval")
    if ask_human(report):
        result = execute("save_report", report)
        status = "done" if result["ok"] else "error"
        print(f"Saved: {result.get('result', result)}")
    else:
        status = "rejected"

    print(f"\nPipeline finished: status={status}, LLM calls=1, "
          f"tokens={tokens}, time={time.time() - start:.1f}s")
    return status


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python pipeline.py "question"')
    run_pipeline(sys.argv[1])
