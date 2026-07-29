# Authorship and AI-assisted implementation

This repository should **not** be read as “one programmer hand-wrote every file solo.”

## Maintainer role

The project maintainer / owner is **not presented here as a professional software engineer by trade**.
His main background is:

- business-process analysis,
- risk management,
- CRM system configuration,
- company architecture and business-process design.

In practice, this means the maintainer's core contribution is:

- defining the problem,
- setting the architecture and constraints,
- writing technical tasks / acceptance criteria,
- reviewing whether the implementation matches the intended logic,
- deciding scope, risks, and claim boundaries.

## How implementation was produced

Implementation in this repository was developed in an **AI-assisted workflow**:

- the maintainer provided architecture, requirements, and review direction;
- AI coding agents / LLM tools generated and edited substantial portions of code and docs;
- outputs were then checked, corrected, narrowed, and accepted or rejected against the maintainer's technical intent.

This is similar in spirit to writing detailed technical tasks for human programmers, then reviewing and iterating on their implementation.

## What is still the maintainer's responsibility

AI assistance does **not** remove responsibility for:

- the claims made in the repository,
- what is marked as tested vs untested,
- the scope of supported model families,
- release decisions,
- risk disclosure.

So the honest reading is:

> architectural direction, task-setting, and acceptance were led by the maintainer;  
> implementation was produced with substantial AI-agent assistance.

## Why this file exists

Open-source and AI-development norms both favor being explicit when code is produced through a white-coding / AI-assisted workflow, especially when readers might otherwise assume a traditional single-author hand-coded process.
