"""
models.py — Shared model wrappers and tier definitions for QASPER eval.

Supports three model families through one interface:
  - Causal instruct models (Qwen2.5-Instruct, Mistral-Instruct): use the
    model's chat template so it stops cleanly at the turn boundary. This is
    the primary path for the current tiers.
  - Causal base models (no chat template): plain text-generation prompt.
  - Encoder-decoder models (T5, BART): AutoModelForSeq2SeqLM.generate().

Using the chat template for instruct models is important: it adds the proper
generation prompt and the model emits an end-of-turn token, which eliminates
the "runaway boilerplate" problem where causal models keep generating past
the real answer into memorized system-prompt text.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")

from raptor import BaseSummarizationModel, BaseQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_causal_answer(text: str) -> str:
    """
    Safety net for causal models that keep generating past the real answer.
    Truncates at the first paragraph break and at known runaway markers.
    With chat-template models this rarely fires, but it's cheap insurance.
    """
    answer = text.strip()
    for lead in ("Short answer:", "Answer:", "Summary:"):
        if answer.startswith(lead):
            answer = answer[len(lead):].strip()
    answer = answer.split("\n\n")[0].strip()
    for marker in (
        "You are an AI assistant",
        "User will",
        "\nQuestion:",
        "\nContext:",
        "\nAnswer:",
        "<|im_end|>",
        "<|im_start|>",
    ):
        if marker in answer:
            answer = answer.split(marker)[0].strip()
    return answer


class _LocalGenerator:
    """
    Shared loading + generation logic for local models. Detects model type
    once and exposes a generate(system, user) method that does the right thing.
    """

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

    def generate(self, system: str, user: str) -> str:
        """Generate a response. Returns "" on failure."""
        self._ensure_loaded()

        # Encoder-decoder (T5/BART)
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

        # Causal model
        if self._pipeline is not None and self._is_causal:
            try:
                if self._has_chat_template:
                    # Instruct model — use the chat template so it stops cleanly
                    messages: List[dict] = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": user})
                    prompt_text = self._pipeline.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                else:
                    # Base causal model — plain prompt
                    prompt_text = (f"{system}\n\n{user}\n\nAnswer:" if system
                                   else f"{user}\n\nAnswer:")

                result = self._pipeline(
                    prompt_text, max_new_tokens=self.max_new_tokens,
                    do_sample=False, return_full_text=False,
                )
                gen = result[0]["generated_text"].strip()
                return _clean_causal_answer(gen)
            except Exception as exc:
                print(f"  [WARN] generation failed: {exc}", file=sys.stderr)
                return ""

        return ""


# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------

class LocalSummarizationModel(BaseSummarizationModel):
    """Local summarizer — chat template for instruct models, generate() otherwise."""

    SYSTEM = ("You are a summarization assistant for scientific text. Produce a "
              "concise summary that preserves key facts, names, and numerical results. "
              "Output only the summary, nothing else.")

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self._gen = _LocalGenerator(model_name, max_new_tokens=128)

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text:
            return ""
        self._gen.max_new_tokens = min(int(max_tokens), 128)
        user = f"Summarize the following text:\n\n{text}"
        out = self._gen.generate(self.SYSTEM, user)
        if out:
            return out
        # Heuristic fallback
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(sentences[:2]) + ("." if sentences else "")


class LocalQAModel(BaseQAModel):
    """Local QA model — chat template for instruct models, generate() otherwise."""

    SYSTEM = ("You are a question answering portal, answer using only the provided context. "
          "Give the most specific answer the context supports — a number, entity, list, or "
          "brief phrase. Extract specific values (scores, counts, names) exactly as stated. "
          "For yes/no questions answer 'Yes' or 'No'. Only reply 'Unanswerable' if the "
          "context genuinely does not contain the answer.")

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 64):
        self._gen = _LocalGenerator(model_name, max_new_tokens=max_new_tokens)

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return ""
        user = f"Context: {context}\n\nQuestion: {question}"
        return self._gen.generate(self.SYSTEM, user)


# ---------------------------------------------------------------------------
# OpenAI API wrappers
# ---------------------------------------------------------------------------

class OpenAIQAModel(BaseQAModel):
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 64):
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
                     "content": "You answer questions about scientific papers. Reply with "
                                "the shortest possible answer: a phrase or entity, not a "
                                "sentence. For yes/no questions answer 'Yes' or 'No'. If the "
                                "context does not contain the answer, reply 'Unanswerable'."},
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
# Tier registry
# ---------------------------------------------------------------------------

MODEL_TIERS = {
    "base": {
        "description": "Qwen2.5-1.5B-Instruct (QA + summ) — small, runs on modest GPU/CPU",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summ": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
        "qa": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
    },
    "local-large": {
        "description": "Qwen2.5-3B-Instruct QA + 1.5B summ — medium",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summ": ("local", "Qwen/Qwen2.5-1.5B-Instruct"),
        "qa": ("local", "Qwen/Qwen2.5-3B-Instruct"),
    },
    "local-xl": {
        "description": "Qwen2.5-7B-Instruct QA + 3B summ — large, needs GPU",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summ": ("local", "Qwen/Qwen2.5-3B-Instruct"),
        "qa": ("local", "Qwen/Qwen2.5-7B-Instruct"),
    },
    "mistral": {
        "description": "Mistral-7B-Instruct QA + Qwen2.5-3B summ — needs GPU + accelerate",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summ": ("local", "Qwen/Qwen2.5-3B-Instruct"),
        "qa": ("local", "mistralai/Mistral-7B-Instruct-v0.3"),
    },
    "api": {
        "description": "GPT-4o-mini via OpenAI API — needs OPENAI_API_KEY",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summ": ("api", "gpt-4o-mini"),
        "qa": ("api", "gpt-4o-mini"),
    },
    "api-gpt4": {
        "description": "GPT-4o via OpenAI API — highest quality and cost",
        "emb": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summ": ("api", "gpt-4o"),
        "qa": ("api", "gpt-4o"),
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