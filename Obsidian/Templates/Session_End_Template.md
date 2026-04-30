# Session End Template

> **When to use:** At the end of every LLM work session. Copy this template, ask the LLM to fill it out, then paste the filled version into `05_SESSION_LOG.md`.

---

## Prompt to Give the LLM

Copy-paste this at the end of your session:

```
Before we end, fill out this session log entry. Be precise — include exact file paths, shapes, and numbers. Do not summarize vaguely.

### Session [NUMBER] — [DATE] — [SHORT TITLE]
**Tool used:** [which LLM/tool]
**Duration:** [approximate]

**What was done:**
- [Bullet list of concrete accomplishments]

**Files created or modified:**
- [exact file paths with shapes/sizes where applicable]

**Decisions made:**
- [Any architecture or design choices, with rationale]

**Issues encountered:**
- [Bugs, unexpected behavior, workarounds]

**New warnings discovered:**
- [Anything that should be added to 03_WARNINGS.md]

**Current blocker:** [What's blocking progress right now, if anything]

**Exact next action:** [The very first thing the next session should do — be specific]
```

---

## After the LLM Fills This Out

1. Copy the filled entry
2. Paste it at the bottom of `05_SESSION_LOG.md`
3. Update `02_PIPELINE_STATUS.md` if any phase status changed
4. Update `04_NEXT_STEPS.md` if the next task changed
5. If any new warnings → add to `03_WARNINGS.md`
6. Re-upload changed files to Perplexity Space (if using Spaces)

**This should take under 3 minutes. Do not skip it.**
