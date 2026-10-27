# Project Execution Rules & Guidelines

These rules are mandatory and govern all work on this project.

---

## 1. Strict Adherence to the Skill File (No Overworking)
* All project work must strictly follow [SKILL_updated.md](file:///e:/PG-Academic/MINI%20-%20PROJECT/EEG-Based%20ADHD%20Analysis/SKILL_updated.md).
* **No overworking**: Do not build unnecessary features, unsolicited deep learning models, complex cloud architectures, or external web apps before the core scientific pipeline is complete. Keep the code focused, clean, and directly tied to the research questions.

## 2. Phase-by-Phase Execution
* Execution must strictly proceed **one phase at a time**.
* Never jump ahead or start modeling before preprocessing and feature extraction are verified.
* Do not move to the next phase until the current phase is fully executed, validated, and approved.

## 3. Mandatory Phase Documentation in `documents/`
* After finishing every phase, create a clear completion document inside the `documents/` folder (e.g., `documents/phase_1_dataset_audit.txt`, `documents/phase_2_preprocessing.txt`, etc.).
* Each document must summarize the implementation details, parameters used, outputs, and findings.

## 4. Reading Hierarchy Before Any Task
At the start of every session or task, always follow this strict sequence:
1. **First**: Read this file (`rules.md`).
2. **Second**: Read [SKILL_updated.md](file:///e:/PG-Academic/MINI%20-%20PROJECT/EEG-Based%20ADHD%20Analysis/SKILL_updated.md).
3. **Third**: Identify and review the specific phase currently being worked on.

## 5. Zero Assumptions & Explicit Clarification
* **Do not assume anything.** If requirements, parameters, or configurations are ambiguous, ask the user directly before making decisions.
* When in doubt, ask immediately.

## 6. Git Push Protocol (Ask Before Pushing)
* **Never push anything to GitHub without explicit user agreement.**
* You may commit locally if appropriate, but you must ask and receive permission before running `git push`.
