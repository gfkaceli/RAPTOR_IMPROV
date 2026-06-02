"""
eval_qasper — QASPER evaluation harness for RAPTOR clustering ablation.

This subpackage runs the 7 clustering methods (flat baseline + 6 RAPTOR
variants) on the QASPER dataset and reports Answer F1 broken down by
answer type (extractive / abstractive / yes-no / unanswerable).

Workflow:
    1. python -m eval_qasper.preprocess_qasper      (one-time, downloads data)
    2. python -m eval_qasper.run_qasper_eval        (builds trees, answers questions)
    3. python -m eval_qasper.score_qasper <run_dir> (computes F1 metrics)
"""
