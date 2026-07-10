"""
eval_quality — QuALITY multiple-choice evaluation harness for the RAPTOR
clustering ablation.

QuALITY (Pang et al., 2022) tests long-passage reading comprehension with
four-option multiple-choice questions. Unlike QASPER, the task is classification
(accuracy), the passages are ~5k tokens (deeper trees), and the dataset ships a
HARD subset of questions that require whole-passage understanding — the case
where hierarchical retrieval is hypothesized to help most.

Workflow:
    1. python -m eval_quality.preprocess_quality --input <QuALITY dev jsonl>
    2. python -m eval_quality.run_quality_eval --model-tier base --max-articles 15
    3. python -m eval_quality.score_quality experiments/quality/<timestamp>
"""
