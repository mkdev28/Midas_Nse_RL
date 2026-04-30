# Perplexity Space — Setup Guide

> **Purpose:** How to configure your Perplexity Pro Space so every new conversation auto-loads project context.

---

## Step 1 — Create the Space

1. Go to [perplexity.ai](https://perplexity.ai)
2. Click **Spaces** in the left sidebar (or go to perplexity.ai/spaces)
3. Click **Create Space**
4. Name it: `MIDAS-NSE`
5. Description: `RL portfolio optimization for NSE NIFTY 50`

---

## Step 2 — Upload These Files

Upload these 4 files from the `Obsidian/` folder into the Space:

| File | Why |
|------|-----|
| `01_LOCKED_ARCHITECTURE.md` | So the LLM knows the system design and cannot deviate |
| `02_PIPELINE_STATUS.md` | So the LLM knows exactly what's done and what's not |
| `03_WARNINGS.md` | So the LLM doesn't repeat known mistakes |
| `04_NEXT_STEPS.md` | So the LLM knows what to work on |

> **Do NOT upload** `00_MASTER_CONTEXT.md` (it's redundant in Spaces — that file is for non-Perplexity LLMs where you paste manually).
> **Do NOT upload** `05_SESSION_LOG.md` (too long, uses up context window on history).

---

## Step 3 — Set the Custom System Prompt

In the Space settings, paste this as the **custom instructions**:

```
You are assisting with MIDAS-NSE, a multi-agent Deep Reinforcement Learning project for NSE portfolio optimization.

MANDATORY RULES:
1. Read all uploaded files before responding to any request.
2. The architecture in 01_LOCKED_ARCHITECTURE.md is FINAL. Do not suggest changes to agent count, state dimensions, reward functions, or the Transformer config unless I explicitly ask.
3. Check 03_WARNINGS.md before writing any code. Violating a warning is a critical error.
4. Always refer to 04_NEXT_STEPS.md to understand what phase we are currently in.
5. When writing code, use paths relative to project root: C:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse\
6. Python 3.12, Windows PowerShell, RTX 4060 8GB VRAM.
7. All data must be pre-computed to parquet/npy. Zero API calls during env.step().
8. At the end of each session, I will ask you to fill out a session log. Be precise with file paths, shapes, and decisions.
```

---

## Step 4 — Re-uploading After Updates

After a work session:
1. Update the relevant Obsidian notes (usually `02_PIPELINE_STATUS.md` and `04_NEXT_STEPS.md`)
2. In Perplexity Space, **delete the old version** of the changed file
3. **Upload the updated version**
4. The next conversation in the Space will use the new file

> **Tip:** You only need to re-upload files that actually changed. Most sessions only change `02` and `04`.

---

## Using the Space

Every new conversation in the `MIDAS-NSE` Space will automatically:
- Load all 4 uploaded files
- Apply the custom system prompt
- Start with full project context

Just open a new thread in the Space and start working. No pasting required.

---

## When Using Non-Perplexity LLMs (Claude, GPT, etc.)

If you need to use a different LLM:
1. Open `00_MASTER_CONTEXT.md` from Obsidian
2. Copy the entire file
3. Paste it as the first message in the new chat
4. Optionally also paste `03_WARNINGS.md` and `04_NEXT_STEPS.md` if context window allows

This gives the non-Perplexity LLM equivalent context to what the Space provides automatically.
