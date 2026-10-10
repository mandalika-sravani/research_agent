"""
Research agent: ReAct loop on the Claude API.

Concepts covered (search the code for the tags):
  [ARCH]       perception -> reasoning -> action -> state, in a control loop
  [REACT]      thought -> action -> observation
  [PLAN]       task decomposition before acting
  [STATE]      state object + checkpoint after every step + resume
  [MEMORY]     episodic memory = message history; semantic = docs/
  [REFLECT]    critique the draft report before it is saved
  [HITL]       approval gate for critical tools
  [GUARD]      max iterations, token budget, loop detection, least privilege

Run:   python agent.py "your research question"
Resume:python agent.py --resume <run_id>
"""
import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder into environment variables

from tools import TOOLS, execute

MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
CKPT_DIR = Path(__file__).parent / "checkpoints"

# [GUARD] hard limits
MAX_STEPS = int(os.getenv("MAX_STEPS", 12))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", 60_000))
MAX_REPEAT_CALLS = int(os.getenv("MAX_REPEAT_CALLS", 2))   # same tool + same args more than this = loop
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", 1))         # reflection rounds

SYSTEM = """You are a careful research agent.

Process:
1. PLAN: Start by writing a short numbered plan (2-5 sub-questions).
2. For each step, briefly say what you're thinking, then call ONE tool.
3. Prefer search_documents for internal topics; use web_search for public/current info.
4. When you have enough evidence, call save_report.
   If save_report comes back with review notes or human feedback, revise the
   report and call save_report again. Never finish without a saved report
   unless a human rejected it.
   Only cite sources you actually retrieved. Never invent URLs.
If a tool returns an error, adjust your approach instead of repeating the same call."""


# ------------------------------------------------------------------ [STATE]
@dataclass
class AgentState:
    run_id: str
    question: str
    messages: list = field(default_factory=list)   # [MEMORY] episodic
    step: int = 0
    tokens_used: int = 0
    revisions: int = 0
    call_counts: dict = field(default_factory=dict)
    status: str = "running"            # running | done | stopped | rejected | no_report

    def save(self):
        CKPT_DIR.mkdir(exist_ok=True)
        (CKPT_DIR / f"{self.run_id}.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, run_id):
        return cls(**json.loads((CKPT_DIR / f"{run_id}.json").read_text()))


def to_dicts(content):
    """Turn SDK content blocks into plain dicts (JSON-checkpointable)."""
    out = []
    for b in content:
        if b.type == "thinking":            # newer models may "think" first; the API wants
            out.append({"type": "thinking",  # these blocks sent back unchanged with tool results
                        "thinking": b.thinking, "signature": b.signature})
        elif b.type == "redacted_thinking":
            out.append({"type": "redacted_thinking", "data": b.data})
        elif b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


# ------------------------------------------------------------------ [HITL]
def ask_human(name, args):
    print("\n" + "=" * 60)
    print(f"APPROVAL NEEDED: agent wants to run '{name}'")
    print(json.dumps(args, indent=2)[:2000])
    print("=" * 60)
    choice = input("[a]pprove / [r]eject / [f]eedback (send notes back): ").strip().lower()
    if choice.startswith("a"):
        return "approve", None
    if choice.startswith("f"):
        return "feedback", input("Your feedback: ")
    return "reject", None


# ------------------------------------------------------------------ [REFLECT]
def reflect(client, state, report):
    """Ask the model to critique its own report against the evidence."""
    evidence = json.dumps(state.messages[-12:])[:15000]
    resp = client.messages.create(
        model=MODEL, max_tokens=2000,   # room for thinking + critique
        messages=[{"role": "user", "content":
            f"Research question: {state.question}\n\nDraft report:\n{json.dumps(report, indent=2)}\n\n"
            f"Evidence gathered (tool results):\n{evidence}\n\n"
            "Critique the draft. Check: does it answer the question? Is every finding supported "
            "by the evidence? Are sources real ones from the evidence?\n"
            "Reply with first line exactly 'APPROVE' or 'REVISE', then 1-4 bullet points."}],
    )
    state.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
    # take only the text blocks (skip any thinking blocks)
    text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:                       # model returned no critique: don't block on nothing
        return True, "(reviewer gave no critique - passing to human approval)"
    return text.upper().startswith("APPROVE"), text


# ------------------------------------------------------------------ [ARCH] control loop
def run(state: AgentState):
    client = Anthropic()  # reads ANTHROPIC_API_KEY
    tool_schemas = [t["schema"] for t in TOOLS.values()]   # [GUARD] least privilege: only these 3

    if not state.messages:
        state.messages.append({"role": "user", "content": state.question})

    # [STATE] resume repair: if we were interrupted after the model asked for a tool
    # but before its result was saved, finish those pending tool calls first.
    last = state.messages[-1]
    if last["role"] == "assistant" and any(b["type"] == "tool_use" for b in last["content"]):
        print("↩️  Finishing the tool call that was interrupted...")
        run_tools(client, state, last["content"])

    while state.status == "running":
        # [GUARD] iteration + budget checks
        if state.step >= MAX_STEPS:
            state.status = "stopped"; print("\n⛔ Max steps reached."); break
        if state.tokens_used >= TOKEN_BUDGET:
            state.status = "stopped"; print("\n⛔ Token budget exhausted."); break

        state.step += 1
        print(f"\n--- step {state.step}  (tokens so far: {state.tokens_used}) ---")

        # [REACT] reasoning
        resp = client.messages.create(model=MODEL, max_tokens=2000, system=SYSTEM,
                                      tools=tool_schemas, messages=state.messages)
        state.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
        state.messages.append({"role": "assistant", "content": to_dicts(resp.content)})

        for b in resp.content:
            if b.type == "text" and b.text.strip():
                print(f"💭 Thought: {b.text.strip()}")

        if resp.stop_reason != "tool_use":
            print("\n(agent stopped without saving a report)")
            state.status = "no_report"; state.save(); break

        # [REACT] actions -> observations
        run_tools(client, state, state.messages[-1]["content"])

    state.save()
    print(f"\nRun {state.run_id} finished with status: {state.status}")


def run_tools(client, state, blocks):
    """[REACT] Run every tool_use block, then append all observations in ONE message."""
    results = []
    for b in blocks:
        if b["type"] != "tool_use":
            continue
        print(f"🔧 Action: {b['name']}({json.dumps(b['input'])[:150]})")
        obs = handle_tool(client, state, b["name"], b["input"])
        print(f"👁  Observation: {json.dumps(obs)[:300]}")
        results.append({"type": "tool_result", "tool_use_id": b["id"],
                        "content": json.dumps(obs)[:8000],
                        "is_error": not obs.get("ok", True)})
    state.messages.append({"role": "user", "content": results})
    state.save()                                       # [STATE] checkpoint


def handle_tool(client, state, name, args):
    # [GUARD] infinite-loop detection
    key = f"{name}:{json.dumps(args, sort_keys=True)}"
    state.call_counts[key] = state.call_counts.get(key, 0) + 1
    if state.call_counts[key] > MAX_REPEAT_CALLS:
        return {"ok": False, "error": "You already made this exact call. Try a different query or finish."}

    if name == "save_report":
        # [REFLECT] verification step before the critical action
        if state.revisions < MAX_REVISIONS:
            ok, critique = reflect(client, state, args)
            print(f"🪞 Reflection:\n{critique}")
            if not ok:
                state.revisions += 1
                return {"ok": False, "error": f"Self-review asked for revision:\n{critique}"}

        # [HITL] approval gate
        if TOOLS[name]["critical"]:
            decision, feedback = ask_human(name, args)
            if decision == "reject":
                state.status = "rejected"
                return {"ok": False, "error": "Human rejected the action."}
            if decision == "feedback":
                return {"ok": False, "error": (
                    f"Not saved yet. The human reviewer left this feedback: '{feedback}'. "
                    "Apply it (if it is only praise, keep the report as is) and call save_report again; "
                    "the human will be asked to approve once more.")}

        obs = execute(name, args)
        if obs["ok"]:
            state.status = "done"
        return obs

    return execute(name, args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--resume")
    a = p.parse_args()
    if a.resume:
        st = AgentState.load(a.resume)
        st.status = "running"
        print(f"Resuming {st.run_id} at step {st.step}")
    elif a.question:
        st = AgentState(run_id=uuid.uuid4().hex[:8], question=a.question)
        print(f"Run id: {st.run_id}")
    else:
        sys.exit('Usage: python agent.py "question"   or   --resume <run_id>')
    try:
        run(st)
    except KeyboardInterrupt:
        st.save()
        print(f"\nPaused. Resume with: python agent.py --resume {st.run_id}")
