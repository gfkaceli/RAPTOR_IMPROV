"""
models.py — Model wrappers and tier definitions for the NarrativeQA experiment.

Self-contained counterpart to eval_qasper/models.py and eval_quality/models.py,
sharing the same generation machinery (chat-template handling for Qwen/Mistral,
seq2seq fallback, cleaning split) with two task-specific choices:

  1. Short free-form QA. NarrativeQA gold answers average roughly five words,
     so the QA prompt demands a brief phrase and the token budget is small.
     There is NO "Unanswerable" label in NarrativeQA — every question has two
     human reference answers — so the prompt forbids refusals and asks for the
     best supported short answer; a refusal would simply score zero.

  2. Narrative summarization. Stories and scripts, same rationale as QuALITY:
     parent nodes must preserve events, characters, relationships, motivations,
     and themes, not scientific "facts and numbers".

EMBEDDING MODEL CONSISTENCY (read this before running):
  Cross-experiment comparison in the thesis requires ONE embedding model across
  QASPER, QuALITY, and NarrativeQA. The default below follows the thesis text;
  if your completed QASPER/QuALITY runs used a different encoder, set
      export RAPTOR_EMB_MODEL="<the model those runs actually used>"
  before running. The resolved value is printed at load time and recorded in
  run_meta.json so the methods section can state it with confidence.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import transformers

os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")

from raptor import BaseSummarizationModel, BaseQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel

# One place to change it; env var wins. MUST match the QASPER/QuALITY runs.
EMB_MODEL = os.environ.get(
    "RAPTOR_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

transformers.set_seed(42)


# ---------------------------------------------------------------------------
# Cleaning helpers (identical behaviour to the other harnesses)
# ---------------------------------------------------------------------------

def _clean_causal_answer(text: str) -> str:
    answer = text.strip()
    for lead in ("Short answer:", "Answer:", "The answer is"):
        if answer.startswith(lead):
            answer = answer[len(lead):].strip()
    answer = answer.split("\n\n")[0].strip()
    for marker in ("You are an AI assistant", "User will", "\nQuestion:",
                   "\nContext:", "\nAnswer:", "<|im_end|>", "<|im_start|>"):
        if marker in answer:
            answer = answer.split(marker)[0].strip()
    return answer


def _clean_summary(text: str) -> str:
    out = text.strip()
    for lead in ("Summary:", "Here is a summary:", "Here's a summary:"):
        if out.startswith(lead):
            out = out[len(lead):].strip()
    for token in ("<|im_end|>", "<|im_start|>"):
        out = out.replace(token, "")
    for marker in ("You are an AI assistant", "\nUser:", "\nSystem:"):
        if marker in out:
            out = out.split(marker)[0].strip()
    return out.strip()


class _LocalGenerator:
    """Shared loader + generator; same logic as the QASPER/QuALITY harnesses."""

    def __init__(self, model_name: str, max_new_tokens: int):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._is_causal = None
        self._has_chat_template = False
        self._load_error = None

    def _ensure_loaded(self):
        if (self._pipeline is not None or self._model is not None
                or self._load_error is not None):
            return
        try:
            from transformers import AutoConfig, AutoTokenizer
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self._is_causal = not (
                hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder
            )
            if self._is_causal:
                from transformers import pipeline as hf_pipeline
                self._pipeline = hf_pipeline(
                    "text-generation", model=self.model_name, tokenizer=self.model_name,
                    trust_remote_code=True, device_map="auto",
                )
                tok = self._pipeline.tokenizer
                self._has_chat_template = getattr(tok, "chat_template", None) is not None
            else:
                from transformers import AutoModelForSeq2SeqLM
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, trust_remote_code=True)
        except Exception as exc:
            self._load_error = exc
            print(f"  [WARN] model load failed ({self.model_name}): {exc}", file=sys.stderr)

    def generate(self, system: str, user: str, clean_mode: str = "answer") -> str:
        self._ensure_loaded()
        cleaner = _clean_summary if clean_mode == "summary" else _clean_causal_answer

        if self._model is not None and not self._is_causal:
            try:
                prompt = f"{system}\n\n{user}" if system else user
                inputs = self._tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1024)
                outputs = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            except Exception:
                return ""

        if self._pipeline is not None and self._is_causal:
            try:
                if self._has_chat_template:
                    messages: List[dict] = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": user})
                    prompt_text = self._pipeline.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                else:
                    prompt_text = (f"{system}\n\n{user}\n\nAnswer:" if system
                                   else f"{user}\n\nAnswer:")
                result = self._pipeline(
                    prompt_text, max_new_tokens=self.max_new_tokens,
                    do_sample=False, return_full_text=False,
                )
                return cleaner(result[0]["generated_text"].strip())
            except Exception as exc:
                print(f"  [WARN] generation failed: {exc}", file=sys.stderr)
                return ""

        return ""


# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------

class LocalSummarizationModel(BaseSummarizationModel):
    """Narrative summarizer — parent nodes for stories and scripts."""

    SYSTEM = (
        "You are summarizing an excerpt from a story or script. Produce a concise "
        "summary that preserves the main events, characters, relationships, "
        "motivations, and themes, not only surface facts. Output only the summary."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self._gen = _LocalGenerator(model_name, max_new_tokens=2048)

    def summarize(self, context, max_tokens=2048):
        text = " ".join(str(context).split())
        if not text:
            return ""
        self._gen.max_new_tokens = min(int(max_tokens), 2048)
        user = f"Summarize the following text:\n\n{text}"
        out = self._gen.generate(self.SYSTEM, user, clean_mode="summary")

        if len(out.split()) < 5:
            retry_user = (
                "Write a summary of the following narrative "
                f"text, preserving events, characters, and themes:\n\n{text}"
            )
            retry = self._gen.generate(self.SYSTEM, retry_user, clean_mode="summary")
            if len(retry.split()) > len(out.split()):
                out = retry

        if len(out.split()) < 3:
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            out = ". ".join(sentences[:2]) + ("." if sentences else "")
            print(f"  [WARN] summary degenerated; used lead-sentence fallback for a "
                  f"{len(text.split())}-word cluster", file=sys.stderr)
        return out


class LocalQAModel(BaseQAModel):
    """
    Short free-form QA for NarrativeQA. Answers are brief phrases; there is no
    unanswerable option, so the prompt requires the best supported short answer.
    """

    QA_SYSTEM = (
        "You answer questions about a story using only the provided excerpts. "
        "Answer with a short phrase, typically under ten words — a name, place, "
        "event, or brief description. Do not explain or refuse; give the best "
        "short answer supported by the excerpts."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 256):
        self._gen = _LocalGenerator(model_name, max_new_tokens=max_new_tokens)

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return ""
        self._gen.max_new_tokens = 256
        user = f"Story excerpts: {context}\n\nQuestion: {question}\n\nShort answer:"
        return self._gen.generate(self.QA_SYSTEM, user, clean_mode="answer")


# ---------------------------------------------------------------------------
# OpenAI API wrappers
# ---------------------------------------------------------------------------

class OpenAIQAModel(BaseQAModel):
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 48):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return ""
        self._ensure_client()
        try:
            r = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "Answer with a short phrase (under ten words) using "
                                "only the provided story excerpts. Never refuse."},
                    {"role": "user",
                     "content": f"Story excerpts: {context}\n\nQuestion: {question}"},
                ],
                max_tokens=self.max_tokens, temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  [WARN] OpenAI QA error: {exc}", file=sys.stderr)
            return ""


class OpenAISummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 150):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text:
            return ""
        self._ensure_client()
        try:
            r = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "Summarize this excerpt from a story or script "
                                "concisely, preserving the main events, characters, "
                                "relationships, motivations, and themes."},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.max_tokens, temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return text[:200]


# ---------------------------------------------------------------------------
# Tier registry — models match the other harnesses; emb resolved above
# ---------------------------------------------------------------------------

MODEL_TIERS = {
    "base": {
        "description": "Qwen2.5-1.5B-Instruct (QA + summ) — small, runs on modest GPU/CPU",
        "emb": EMB_MODEL,
        "summ": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
        "qa": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
    },
    "local-large": {
        "description": "Qwen2.5-3B-Instruct QA + 1.5B summ — medium",
        "emb": EMB_MODEL,
        "summ": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
        "qa": ("local", "Qwen/Qwen2.5-3B-Instruct"),
    },
    "local-xl": {
        "description": "Qwen2.5-7B-Instruct QA + 3B summ — large, needs GPU",
        "emb": EMB_MODEL,
        "summ": ("local", "Qwen/Qwen2.5-3B-Instruct"),
        "qa": ("local", "Qwen/Qwen2.5-7B-Instruct"),
    },
    "mistral": {
        "description": "Mistral-7B-Instruct QA + Qwen2.5-3B summ — needs GPU + accelerate",
        "emb": EMB_MODEL,
        "summ": ("local", "Qwen/Qwen2.5-3B-Instruct"),
        "qa": ("local", "mistralai/Mistral-7B-Instruct-v0.3"),
    },
    "api": {
        "description": "GPT-4o-mini via OpenAI API — needs OPENAI_API_KEY",
        "emb": EMB_MODEL,
        "summ": ("api", "gpt-4o-mini"),
        "qa": ("api", "gpt-4o-mini"),
    },
    "api-gpt4": {
        "description": "GPT-4o via OpenAI API — highest quality and cost",
        "emb": EMB_MODEL,
        "summ": ("api", "gpt-4o"),
        "qa": ("api", "gpt-4o"),
    },
}


def load_models(tier_name: str = "base") -> Tuple:
    """Build (embedding_model, summarization_model, qa_model) for the tier."""
    if tier_name not in MODEL_TIERS:
        raise ValueError(f"Unknown tier '{tier_name}'. Choices: {list(MODEL_TIERS.keys())}")
    tier = MODEL_TIERS[tier_name]
    print(f"  Tier: {tier_name} — {tier['description']}")
    print(f"  Embedding model (MUST match QASPER/QuALITY runs): {tier['emb']}")

    emb = SBertEmbeddingModel(model_name=tier["emb"])

    summ_type, summ_name = tier["summ"]
    summ = (OpenAISummarizationModel(model_name=summ_name) if summ_type == "api"
            else LocalSummarizationModel(model_name=summ_name))

    qa_type, qa_name = tier["qa"]
    qa = (OpenAIQAModel(model_name=qa_name) if qa_type == "api"
          else LocalQAModel(model_name=qa_name))

    return emb, summ, qa
