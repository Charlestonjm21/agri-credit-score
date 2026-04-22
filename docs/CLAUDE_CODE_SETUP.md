# Setting up this project in Claude Code

A quickstart for working on agri-credit-score with Claude Code on WSL/Ubuntu.

## Prerequisites (you likely have most of these from dse-ml-investor)

- WSL2 with Ubuntu
- Node.js 18+ installed (Claude Code is an npm package)
- Python 3.10+
- Git with PAT authentication to GitHub
- A Claude.ai account

## One-time install

Follow the official Anthropic install instructions at https://docs.claude.com/en/docs/claude-code/overview — they're the source of truth for the current install command. The docs cover WSL/Linux setup including any Node version requirements and authentication flow.

## Project setup

From your WSL home or projects directory:

```bash
# 1. Unzip the scaffold
unzip agri-credit-score.zip
cd agri-credit-score

# 2. Initialize git
git init
git add .
git commit -m "Initial scaffold: v0 synthetic-data credit scoring"

# 3. Create a GitHub repo and push
# (Use the same PAT flow you set up for dse-ml-investor)
git branch -M main
git remote add origin https://github.com/charlestonjm21/agri-credit-score.git
git push -u origin main

# 4. Set up a Python venv (recommended over system Python)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 5. Smoke-test the pipeline
make data
make train
make test
```

If `make test` passes 22/22, you're ready to open Claude Code.

## Launching Claude Code in the project

```bash
cd ~/agri-credit-score  # or wherever you put it
claude
```

On first launch in this directory, Claude Code will read `CLAUDE.md` automatically. You don't need to paste context — the CLAUDE.md file is written so Claude understands the project, conventions, and current state immediately.

## Good first prompts for Claude Code

Once inside, these are productive starter prompts that match the project's next priorities:

1. **"Read CLAUDE.md and docs/SPEC.md, then add a time-based holdout evaluation mode to models/train.py. It should train on months 1–18 of simulated data and test on months 19–24 to simulate the retrospective backtest we'll run with a real lender."**

2. **"Draft docs/WORKING_PAPER.md as an 8–12 page technical report following the structure in the SPEC's definition-of-done. Sections: abstract, introduction, related work on East African alternative-data credit scoring, methodology, synthetic data design, results, fairness audit, limitations, future work with real data."**

3. **"Review the Swahili summary in README.md for accuracy and idiomatic phrasing. If anything needs improvement, propose revisions inline."**

4. **"Add a Dockerfile and docker-compose.yml that run the FastAPI service. The image should be slim, not include dev dependencies, and expose port 8000."**

5. **"Add API key auth to the FastAPI service as middleware. Keys should be loaded from an environment variable. Don't break any existing tests; update tests to use a test key."**

## Habits to keep the project healthy

- **Ask Claude Code to run `make test` before committing.** It will. One command, 22 tests, catches regressions fast.
- **Update CLAUDE.md as the project evolves.** When you add a new dependency, change a convention, or move to Phase 2 of the SPEC, update CLAUDE.md so future Claude Code sessions have the right context.
- **Commit often with real messages.** `"updates"` is not a commit message. `"Add time-based holdout to train.py; update SPEC §6.2"` is.
- **Keep docs/SPEC.md authoritative.** If Claude Code suggests a design change, ask it to update SPEC.md in the same change.
- **Don't let scope creep in.** If you find yourself prompting "let's also add a mobile app for farmers", stop. CLAUDE.md explicitly lists that as out-of-scope for v0. Stay disciplined.

## When to use Claude Code vs. this chat

- **Use Claude Code for**: multi-file edits, refactors, running tests, iterating on models, debugging tracebacks, writing code-adjacent content (docstrings, the working paper drafted inside the repo).
- **Use this chat for**: strategic decisions, outreach email drafts, interview prep, pivot discussions, reading lists for financial inclusion research, networking plans — anything that isn't code editing.

Both will have memory of your broader goals via the memory system. Claude Code gets project context from CLAUDE.md.
