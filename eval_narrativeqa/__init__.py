"""
eval_narrativeqa — NarrativeQA free-form QA evaluation harness for the RAPTOR
clustering ablation (Experiment #3 in the thesis benchmark gradient).

NarrativeQA (Kocisky et al., 2018) asks questions over full books and movie
scripts — documents ~10x longer than QuALITY passages. It is the benchmark
where the thesis hypothesis predicts the LARGEST hierarchy gain: questions
were written from plot summaries, answers are short free-form paraphrases
(two references each, max-over-references scoring), and ROUGE-L is the
headline metric.

Workflow:
    1. python -m eval_narrativeqa.preprocess_narrativeqa --split validation --max-articles 5
    2. python -m eval_narrativeqa.run_narrativeqa_eval --model-tier base \
           --cache-dir <drive path>   # tree caching is essential at this scale
    3. python -m eval_narrativeqa.score_narrativeqa experiments/narrativeqa/<timestamp>
"""
