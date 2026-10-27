---
name: eeg-adhd-reduced-montage
description: Guides work on the project "ADHD Analysis Using EEG: Testing Accuracy with Fewer Channels and Noise" (comparing machine-learning performance with all usable EEG channels versus a predefined smaller channel set, and testing performance under controlled simulated noise; optional missing-channel and SHAP-stability experiments). Use whenever the user mentions EEG, ADHD classification, band power, theta/beta ratio, Welch PSD, MNE-Python, channel selection, subject-independent or participant-wise splitting, EEG noise robustness, or asks for code, review, report text, or slides for this project, even if they do not name the project explicitly. Also trigger on data leakage, reduced montage, or noise sensitivity in EEG models.
---

# ADHD Analysis Using EEG: Testing Accuracy with Fewer Channels and Noise

Helps the user build a reproducible, honest research prototype that answers two clear questions: (1) Can machine-learning models analyse EEG data using fewer channels? (2) How does noise affect their accuracy? Optional extensions examine missing channels and the stability of model explanations.

This is a research prototype, **not a diagnostic tool**. The reasons behind each rule are given so you can apply judgement on cases not listed here.

## Core principles

1. **Verify before assuming.** Details of the planned dataset (the Nasrabadi pediatric ADHD EEG set, often cited as 121 children with 19 channels) come from secondary literature. Confirm participant count, labels, channel names, sampling frequency, units, file format, and licence from the real files and official documentation before designing experiments. If metadata is insufficient, stop and propose a documented alternative dataset.
2. **No fabricated results.** Never invent metrics, participants, citations, or findings. If an experiment has not been run, say so and offer to write the code that runs it. Report negative results (for example, reduced channels performing worse) as valid findings.
3. **No diagnostic claims.** Treat dataset labels as research labels. Band power and theta/beta ratio (TBR) are exploratory features, never standalone markers. Do not say a model diagnoses, rules out, or treats ADHD.
4. **Leakage is the main way this project fails.** A random split of EEG segments mixes one child's signal into both train and test sets and inflates scores, so a result built that way is not subject-independent.

## Workflow

Follow this order. Do not move on until the earlier milestone works.

1. **Inspect data.** List files, build a participant-level metadata table, confirm labels and channels, plot one recording, write `dataset_audit.md` with problems and exclusions. Keep raw data unchanged.
2. **Preprocess.** Confirm sampling rate and units, choose filters from the actual recording characteristics, document every exclusion with a reason, and plot raw versus processed signals.
3. **Extract features.** Welch PSD per channel; theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz. State whether power is absolute or relative, keep windowing consistent, and check the sampling rate supports the bands. Output a matrix with participant ID, label, and named features.
4. **Split by participant.** Keep all segments from one participant in one split. Use group-aware cross-validation (for example `GroupKFold` or `StratifiedGroupKFold`) for model selection. Fit scaling, feature selection, and tuning on training data only, inside a pipeline. Keep final test participants untouched. If participant IDs are unavailable, document that and do not claim subject-independent evaluation.
5. **Baselines.** Random Forest and SVM with simple, fixed settings; avoid heavy tuning on a small dataset. Report balanced accuracy, sensitivity, specificity, precision, F1, ROC-AUC when valid, confusion matrices, and class counts. Choose metrics before looking at results.
6. **Channel comparison.** All usable channels versus a channel set defined in advance and justified from literature (for example Fz, Cz, F3, F4) only if those names exist and map reliably. Never choose channels using test-set performance.
7. **Noise robustness** (only after a clean baseline works). Keep the original signals untouched, add documented noise with fixed seeds and defined levels to a copy of the test signals, evaluate at each level, and plot performance against noise. Label it a simulation.
8. **Optional: missing channels.** Define which channels are removed and why; re-run the full pipeline. Do not zero-fill without a justified protocol, since it creates artificial patterns.
9. **Optional: SHAP stability.** Define the stability measure before analysing (for example rank correlation of attributions across folds). Describe SHAP as a model explanation, not biological causation.
10. **Final checks.** Fresh-environment run works, seeds set, no test information used for fitting, figures and tables generated from saved outputs, every reported number matches code output, limitations documented.

If the user has only two weeks, prioritise steps 1–6 plus the report and slides, and treat steps 7–9 and any web interface as stretch goals.

## Code conventions

- Python with MNE-Python, SciPy, NumPy, pandas, scikit-learn, Matplotlib; SHAP only if time permits.
- Prefer a simple local pipeline. Do not build a React or FastAPI app, authentication, cloud deployment, or deep-learning models before the core experiment works.
- Suggested layout:

```text
eeg-adhd-research/
├── data/README.md
├── notebooks/        # 01_dataset_exploration, 02_baseline_experiments
├── src/              # load_data, preprocess, extract_features, train_models, evaluate, robustness
├── tests/
├── results/{figures,metrics}/
├── report/
├── requirements.txt
└── README.md
```

- Set random seeds, save metrics to files, and generate figures from those files.
- Never commit restricted or large dataset files; give download instructions in `data/README.md`.
- When writing code, add a small test or assertion for the riskiest assumptions: no participant overlap between train and test, expected channel names present, expected sampling rate.

## Reviewing the user's work

Check in this order and report problems plainly:
1. Is any participant in both train and test? Was any fitting done on test data?
2. Were channels, filters, or hyperparameters chosen after seeing test results?
3. Do reported numbers match saved outputs, and are class counts and uncertainty given?
4. Does any wording imply diagnosis or an unverified novelty claim?

## Writing support

- **Report structure:** Introduction, Literature review, Methodology, Implementation and experiments, Results and discussion, Conclusion and future work, References.
- **Slides:** 10–12 slides covering title, problem, literature, gap, objectives, architecture, methodology, experiments, results, limitations, conclusion, references.
- **Novelty:** Phrase the contribution as a "candidate contribution" until a literature review is done. Reduced-channel classification and SHAP are each already studied. Only the specific combination (subject-independent evaluation, full versus reduced channels, noise or missing-channel robustness, explanation stability) might be underexplored. Record each paper's dataset, split strategy, channels, noise tests, explainability method, and limitations in a comparison table, and suggest the user confirm the narrowed question with their guide.
- **Demo (5–7 min):** problem and objective, verified dataset, signal processing, baseline model and split, research experiment with real plots, conclusion with limitations.
- **Likely guide questions:** why EEG and this dataset, what PSD and TBR are, why not use TBR to diagnose, why Random Forest and SVM, why participant-wise splitting, how channels were chosen, how noise was generated, what the research gap is, what the limitations are.

## Boundaries

- Respect dataset licence, privacy, and citation requirements.
- State class imbalance, sample size, exact evaluation protocol, and limitations in every results section.
- Treat artificial-noise results as controlled simulation, not proof of real-world robustness.

## First task

Verify the dataset and its documentation, then load and plot one recording. Do not start modelling or any frontend until that succeeds.
