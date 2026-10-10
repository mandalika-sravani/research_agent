# Research Agent

A ReAct research agent on the Claude API with web search, local document retrieval,
a structured report (Markdown + JSON), and a human approval step before it saves.

## Setup (Windows)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    (then put your key in .env)
```

## Simple task to start with
```
python agent.py "Which vector database should a student use for a local RAG project, and why?"
```
What you should see:
1. **Plan**: the agent writes 2-5 sub-questions.
2. **Thought → Action → Observation**: it calls `search_documents` (finds `docs/vector-dbs-notes.md`), then `web_search`.
3. **Reflection**: it critiques its own draft and may revise once.
4. **HITL gate**: you are asked to `[a]pprove / [r]eject / [f]eedback`. Try `f` first, e.g. "add a comparison of costs", and watch it revise.
5. Report saved in `reports/`.

Then try the guardrails and resume:
- Press **Ctrl+C** mid-run, then `python agent.py --resume <run_id>`. It continues from the checkpoint.
- Set `MAX_STEPS=2` in `.env` and watch the "Max steps reached" stop.

## Where each concept lives (search for the tag in agent.py)
| Concept | Tag / file |
|---|---|
| Architecture, control loop | `[ARCH]` `run()` |
| ReAct | `[REACT]` |
| Tool schemas, validation, timeout, retries | `tools.py`: `TOOLS`, `validate()`, `execute()` |
| Planning | `[PLAN]` system prompt step 1 |
| State, checkpoints, memory | `[STATE]` `AgentState`, `checkpoints/`; `[MEMORY]` |
| Reflection | `[REFLECT]` `reflect()` |
| Human-in-the-loop | `[HITL]` `ask_human()` |
| Guardrails | `[GUARD]` MAX_STEPS, TOKEN_BUDGET, loop detection, only 3 tools |

## Upgrade ideas
- Replace keyword search in `search_documents` with your Module 6 embeddings + Chroma.
- Add a deterministic version (search → summarize → save, fixed order) and compare it with the agent.
