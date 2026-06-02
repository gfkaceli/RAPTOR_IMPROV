"""
models.py — Shared model wrappers and tier definitions for QASPER eval.

Duplicates the LocalSummarizationModel, LocalQAModel, OpenAIQAModel, and
OpenAISummarizationModel from eval_demo.py. Pulled out into this module so
that the QASPER scripts can import them without depending on the notebook
or eval_demo.py being in a particular location.

If you later refactor eval_demo.py to import from here, that's fine — these
two copies should stay synchronized.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")

from raptor import BaseSummarizationModel, BaseQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel


# ---------------------------------------------------------------------------
# Local models — handle both encoder-decoder (T5/BART) and causal (Mistral)
# ---------------------------------------------------------------------------

class LocalSummarizationModel(BaseSummarizationModel):
    """
    Local summarizer.
    Uses AutoModelForSeq2SeqLM.generate() for encoder-decoder models,
    pipeline("text-generation") for causal models.
    """

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._load_error = None
        self._is_causal = None

    def _ensure_loaded(self):
        if self._model is not None or self._pipeline is not None or self._load_error is not None:
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
            else:
                from transformers import AutoModelForSeq2SeqLM
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as exc:
            self._load_error = exc
            print(f"  [WARN] Summarizer load failed: {exc}", file=sys.stderr)

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text:
            return ""
        self._ensure_loaded()
        max_out = min(int(max_tokens), 128)

        if self._model is not None and not self._is_causal:
            try:
                inputs = self._tokenizer(
                    f"Summarize: {text}", return_tensors="pt",
                    truncation=True, max_length=1024,
                )
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_out, do_sample=False,
                )
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            except Exception:
                pass

        if self._pipeline is not None and self._is_causal:
            prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"
            try:
                result = self._pipeline(prompt, max_new_tokens=max_out, do_sample=False)
                gen = result[0]["generated_text"]
                return (gen.split("Summary:")[-1].strip() if "Summary:" in gen
                        else gen[len(prompt):].strip())
            except Exception:
                pass

        # Heuristic fallback
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(sentences[:2]) + ("." if sentences else "")


class LocalQAModel(BaseQAModel):
    """
    Local QA model.
    Uses AutoModelForSeq2SeqLM.generate() for encoder-decoder models,
    pipeline("text-generation") for causal models.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 80):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._load_error = None
        self._is_causal = None

    def _ensure_loaded(self):
        if self._model is not None or self._pipeline is not None or self._load_error is not None:
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
            else:
                from transformers import AutoModelForSeq2SeqLM
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as exc:
            self._load_error = exc
            print(f"  [WARN] QA model load failed: {exc}", file=sys.stderr)

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return ""
        self._ensure_loaded()

        prompt = (
            f"Answer the question using only the context from a scientific paper below. "
            f"If the question is extractive:"
            f"Reply with the shortest possible answer: a single phrase, entity, or list of entities."
            f"Otherwise reply with 5 to 7 words"
            f"For yes/no questions, answer exactly 'Yes' or 'No'. "
            f"If the context does not contain the answer, reply exactly with 'Unanswerable'.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        if self._model is not None and not self._is_causal:
            try:
                inputs = self._tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=1024,
                )
                outputs = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            except Exception:
                pass

        if self._pipeline is not None and self._is_causal:
            try:
                result = self._pipeline(
                    prompt, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
                gen = result[0]["generated_text"]
                return (gen.split("Answer:")[-1].strip() if "Answer:" in gen
                        else gen[len(prompt):].strip())
            except Exception:
                pass

        return ""


# ---------------------------------------------------------------------------
# OpenAI API wrappers
# ---------------------------------------------------------------------------

class OpenAIQAModel(BaseQAModel):
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 150):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
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
                     "content": "You are a precise question-answering assistant for "
                                "scientific papers. Answer based only on the provided "
                                "context. Be concise. If the context does not contain "
                                "the answer, respond with 'Unanswerable'."},
                    {"role": "user",
                     "content": f"Context: {context}\n\nQuestion: {question}"},
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
        if self._client is not None:
            return
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
                     "content": "Summarize the following text from a scientific paper "
                                "concisely, preserving key facts, names, and numerical results."},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.max_tokens, temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return text[:200]


# ---------------------------------------------------------------------------
# Tier registry — same as eval_demo.py
# ---------------------------------------------------------------------------

MODEL_TIERS = {
    "base":        {
        "description": "QWEN small",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summ": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
        "qa": ("local", "Qwen/Qwen2.5-1.5B-Instruct")
    },
    "local-large": {"description": "qwen medium",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                    "summ": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
                    "qa": ("local", "Qwen/Qwen2.5-3B-Instruct")},
    "local-xl":    {"description": "qwen large",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                    "summ": ("local", "Qwen/Qwen2.5-3B-Instruct"),
                    "qa": ("local", "Qwen/Qwen2.5-7B-Instruct")},
    "mistral":     {"description": "mistral",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                    "summ": ("local", "Qwen/Qwen2.5-3B-Instruct"),
                    "qa": ("local", "mistralai/Mistral-7B-Instruct-v0.3")},
    "api":         {"description": "GPT-4o mini",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                    "summ": ("api",   "gpt-4o-mini"),
                    "qa": ("api",   "gpt-4o-mini")},
    "api-gpt4":    {"description": "GPT-4o",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                    "summ": ("api",   "gpt-4o"),  "qa": ("api", "gpt-4o")
                    },
}


def load_models(tier_name: str = "base") -> Tuple:
    """Build (embedding_model, summarization_model, qa_model) for the given tier."""
    if tier_name not in MODEL_TIERS:
        raise ValueError(f"Unknown tier '{tier_name}'. Choices: {list(MODEL_TIERS.keys())}")
    tier = MODEL_TIERS[tier_name]
    print(f"  Tier: {tier_name} — {tier['description']}")

    emb = SBertEmbeddingModel(model_name=tier["emb"])

    summ_type, summ_name = tier["summ"]
    summ = (OpenAISummarizationModel(model_name=summ_name) if summ_type == "api"
            else LocalSummarizationModel(model_name=summ_name))

    qa_type, qa_name = tier["qa"]
    qa = (OpenAIQAModel(model_name=qa_name) if qa_type == "api"
          else LocalQAModel(model_name=qa_name))

    return emb, summ, qa
