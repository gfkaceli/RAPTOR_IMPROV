"""
models.py — Model wrappers and tier definitions for the QuALITY experiment.

This is a QuALITY-specific counterpart to eval_qasper/models.py. It shares the
same loading and generation machinery (chat-template handling for Qwen/Mistral
instruct models, seq2seq fallback for T5/BART, OpenAI wrappers, and the tier
registry) but differs in two task-specific ways:

  1. Multiple-choice QA. QuALITY is a four-option classification task, not
     free-form generation. LocalQAModel and OpenAIQAModel therefore expose an
     answer_multiple_choice(context, question, options) method that formats the
     options as A/B/C/D, instructs the model to reply with a single letter, and
     caps generation tightly. The plain answer_question(context, question) method
     is retained only for RAPTOR interface compatibility.

  2. Narrative summarization. QuALITY passages are fiction and journalism
     (Project Gutenberg stories, Slate articles), not scientific papers. The
     summarizer's system prompt preserves events, characters, relationships,
     motivations, and themes rather than "facts, names, and numerical results".
     Using the scientific prompt here would under-represent the narrative content
     that the HARD (whole-passage) questions probe, confounding a hierarchy
     failure with a prompt mismatch.

Everything else — the tier list, the seed behaviour, the runaway-answer cleaning
— matches the QASPER harness so results across the two datasets stay comparable.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")

from raptor import BaseSummarizationModel, BaseQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel


LETTERS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Shared cleaning helpers
# ---------------------------------------------------------------------------

def _clean_causal_answer(text: str) -> str:
    """
    Strict cleaning for short answers (including single-letter MC answers).
    Causal models sometimes emit the answer then drift into boilerplate; we
    truncate at the first paragraph break and known runaway markers.
    """
    answer = text.strip()
    for lead in ("Short answer:", "Answer:", "The answer is", "Option", "Choice"):
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


def _clean_summary(text: str) -> str:
    """
    Light cleaning for summaries. Summaries are multi-sentence, so we must NOT
    truncate at the first paragraph break the way answer cleaning does. Strip a
    leading label and chat control tokens; cut only at explicit prompt-echo
    markers.
    """
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
    """
    Shared loading + generation logic. Detects model type once and exposes a
    generate(system, user, clean_mode) method. Identical to the QASPER harness
    so both datasets run through the same generation path.
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

    def generate(self, system: str, user: str, clean_mode: str = "answer") -> str:
        """Generate a response. clean_mode is 'answer' (strict) or 'summary' (light)."""
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
                gen = result[0]["generated_text"].strip()
                return cleaner(gen)
            except Exception as exc:
                print(f"  [WARN] generation failed: {exc}", file=sys.stderr)
                return ""

        return ""


# ---------------------------------------------------------------------------
# Multiple-choice prompt formatting (shared by local and API QA models)
# ---------------------------------------------------------------------------

MC_SYSTEM = (
    "You are answering a multiple-choice reading-comprehension question about a "
    "passage. You are given retrieved excerpts from the passage and a question "
    "with four options labelled A, B, C, and D. Choose the single best option "
    "using only the excerpts. Respond with just the letter (A, B, C, or D) and "
    "nothing else."
)


def format_mc_user(context: str, question: str, options: List[str]) -> str:
    letters = LETTERS[:len(options)]
    opt_lines = "\n".join(f"{L}. {opt}" for L, opt in zip(letters, options))
    return (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_lines}\n\n"
        f"Answer with a single letter ({'/'.join(letters)})."
    )


# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------

class LocalSummarizationModel(BaseSummarizationModel):
    """Local summarizer with a narrative-appropriate system prompt for QuALITY."""

    SYSTEM = (
        "You are summarizing an excerpt from a story or article. Produce a concise "
        "summary that preserves the main events, characters, relationships, "
        "motivations, and themes, not only surface facts. Output only the summary."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self._gen = _LocalGenerator(model_name, max_new_tokens=128)

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text:
            return ""
        self._gen.max_new_tokens = min(int(max_tokens), 128)
        user = f"Summarize the following text:\n\n{text}"
        out = self._gen.generate(self.SYSTEM, user, clean_mode="summary")

        # Guard against near-empty summaries (causal models occasionally emit
        # almost nothing); retry once, then fall back to lead sentences so the
        # node carries real content rather than noise.
        if len(out.split()) < 5:
            retry_user = (
                "Write a concise 2-3 sentence summary of the following narrative "
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
    Local QA model for QuALITY.

    The primary method is answer_multiple_choice(context, question, options),
    which formats the options and asks the model for a single letter. The plain
    answer_question(context, question) is kept for RAPTOR interface compatibility
    (RAPTOR constructs the config with a qa_model), but the QuALITY runner calls
    answer_multiple_choice directly after question-only retrieval.
    """

    QA_SYSTEM = (
        "You answer questions about a passage using only the provided context. "
        "Be concise."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 8):
        # MC answers are a single letter, so the default token budget is tiny.
        self._gen = _LocalGenerator(model_name, max_new_tokens=max_new_tokens)

    def answer_multiple_choice(self, context, question, options) -> str:
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context or not options:
            return ""
        self._gen.max_new_tokens = 8  # single letter
        user = format_mc_user(context, question, options)
        return self._gen.generate(MC_SYSTEM, user, clean_mode="answer")

    def answer_question(self, context, question):
        # RAPTOR-compatibility path (not used for QuALITY scoring).
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return ""
        self._gen.max_new_tokens = 64
        user = f"Context: {context}\n\nQuestion: {question}"
        return self._gen.generate(self.QA_SYSTEM, user, clean_mode="answer")


# ---------------------------------------------------------------------------
# OpenAI API wrappers
# ---------------------------------------------------------------------------

class OpenAIQAModel(BaseQAModel):
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 8):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI()

    def answer_multiple_choice(self, context, question, options) -> str:
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context or not options:
            return ""
        self._ensure_client()
        try:
            r = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": MC_SYSTEM},
                    {"role": "user", "content": format_mc_user(context, question, options)},
                ],
                max_tokens=self.max_tokens, temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  [WARN] OpenAI MC error: {exc}", file=sys.stderr)
            return ""

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
                    {"role": "system", "content": "Answer concisely using only the context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                ],
                max_tokens=64, temperature=0,
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
                     "content": "Summarize this excerpt from a story or article concisely, "
                                "preserving the main events, characters, relationships, "
                                "motivations, and themes."},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.max_tokens, temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return text[:200]


# ---------------------------------------------------------------------------
# Tier registry — same models as the QASPER harness for cross-dataset parity
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
"""
models.py — Model wrappers and tier definitions for the QuALITY experiment.

This is a QuALITY-specific counterpart to eval_qasper/models.py. It shares the
same loading and generation machinery (chat-template handling for Qwen/Mistral
instruct models, seq2seq fallback for T5/BART, OpenAI wrappers, and the tier
registry) but differs in two task-specific ways:

  1. Multiple-choice QA. QuALITY is a four-option classification task, not
     free-form generation. LocalQAModel and OpenAIQAModel therefore expose an
     answer_multiple_choice(context, question, options) method that formats the
     options as A/B/C/D, instructs the model to reply with a single letter, and
     caps generation tightly. The plain answer_question(context, question) method
     is retained only for RAPTOR interface compatibility.

  2. Narrative summarization. QuALITY passages are fiction and journalism
     (Project Gutenberg stories, Slate articles), not scientific papers. The
     summarizer's system prompt preserves events, characters, relationships,
     motivations, and themes rather than "facts, names, and numerical results".
     Using the scientific prompt here would under-represent the narrative content
     that the HARD (whole-passage) questions probe, confounding a hierarchy
     failure with a prompt mismatch.

Everything else — the tier list, the seed behaviour, the runaway-answer cleaning
— matches the QASPER harness so results across the two datasets stay comparable.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")

from raptor import BaseSummarizationModel, BaseQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel

import transformers

transformers.set_seed(42)

LETTERS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Shared cleaning helpers
# ---------------------------------------------------------------------------

def _clean_causal_answer(text: str) -> str:
    """
    Strict cleaning for short answers (including single-letter MC answers).
    Causal models sometimes emit the answer then drift into boilerplate; we
    truncate at the first paragraph break and known runaway markers.
    """
    answer = text.strip()
    for lead in ("Short answer:", "Answer:", "The answer is", "Option", "Choice"):
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


def _clean_summary(text: str) -> str:
    """
    Light cleaning for summaries. Summaries are multi-sentence, so we must NOT
    truncate at the first paragraph break the way answer cleaning does. Strip a
    leading label and chat control tokens; cut only at explicit prompt-echo
    markers.
    """
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
    """
    Shared loading + generation logic. Detects model type once and exposes a
    generate(system, user, clean_mode) method. Identical to the QASPER harness
    so both datasets run through the same generation path.
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

    def generate(self, system: str, user: str, clean_mode: str = "answer") -> str:
        """Generate a response. clean_mode is 'answer' (strict) or 'summary' (light)."""
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
                gen = result[0]["generated_text"].strip()
                return cleaner(gen)
            except Exception as exc:
                print(f"  [WARN] generation failed: {exc}", file=sys.stderr)
                return ""

        return ""


# ---------------------------------------------------------------------------
# Multiple-choice prompt formatting (shared by local and API QA models)
# ---------------------------------------------------------------------------

MC_SYSTEM = (
    "You are answering a multiple-choice question about a passage."
    "You are given text from the passage and a question "
    "with four options labelled A, B, C, and D. Choose the single best option "
    "using only the context."
)


def format_mc_user(context: str, question: str, options: List[str]) -> str:
    letters = LETTERS[:len(options)]
    opt_lines = "\n".join(f"{L}. {opt}" for L, opt in zip(letters, options))
    return (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_lines}\n\n"
        f"Answer with a single letter ({'/'.join(letters)})."
    )


# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------

class LocalSummarizationModel(BaseSummarizationModel):
    """Local summarizer with a narrative-appropriate system prompt for QuALITY."""

    SYSTEM = (
        "You are summarizing an excerpt from a story or article. Produce a summary "
        " that preserves the main events, characters, relationships, "
        "motivations, and themes, not only surface facts. Output only the summary."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self._gen = _LocalGenerator(model_name, max_new_tokens=1024)

    def summarize(self, context, max_tokens=1024):
        text = " ".join(str(context).split())
        if not text:
            return ""
        self._gen.max_new_tokens = min(int(max_tokens), 1024)
        user = f"Summarize the following text:\n\n{text}"
        out = self._gen.generate(self.SYSTEM, user, clean_mode="summary")

        # Guard against near-empty summaries (causal models occasionally emit
        # almost nothing); retry once, then fall back to lead sentences so the
        # node carries real content rather than noise.
        """if len(out.split()) < 5:
            retry_user = (
                "Write a concise summary of the following narrative "
                f"text, preserving events, characters, and themes:\n\n{text}"
            )
            retry = self._gen.generate(self.SYSTEM, retry_user, clean_mode="summary")
            if len(retry.split()) > len(out.split()):
                out = retry

        if len(out.split()) < 3:
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            out = ". ".join(sentences[:2]) + ("." if sentences else "")
            print(f"  [WARN] summary degenerated; used lead-sentence fallback for a "
                  f"{len(text.split())}-word cluster", file=sys.stderr)"""
        return out


class LocalQAModel(BaseQAModel):
    """
    Local QA model for QuALITY.

    The primary method is answer_multiple_choice(context, question, options),
    which formats the options and asks the model for a single letter. The plain
    answer_question(context, question) is kept for RAPTOR interface compatibility
    (RAPTOR constructs the config with a qa_model), but the QuALITY runner calls
    answer_multiple_choice directly after question-only retrieval.
    """

    QA_SYSTEM = (
        "You answer questions about a passage using only the provided context. "
        "Be concise."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 8):
        # MC answers are a single letter, so the default token budget is tiny.
        self._gen = _LocalGenerator(model_name, max_new_tokens=max_new_tokens)

    def answer_multiple_choice(self, context, question, options) -> str:
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context or not options:
            return ""
        self._gen.max_new_tokens = 8  # single letter
        user = format_mc_user(context, question, options)
        return self._gen.generate(MC_SYSTEM, user, clean_mode="answer")

    def answer_question(self, context, question):
        # RAPTOR-compatibility path (not used for QuALITY scoring).
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return ""
        self._gen.max_new_tokens = 64
        user = f"Context: {context}\n\nQuestion: {question}"
        return self._gen.generate(self.QA_SYSTEM, user, clean_mode="answer")


# ---------------------------------------------------------------------------
# OpenAI API wrappers
# ---------------------------------------------------------------------------

class OpenAIQAModel(BaseQAModel):
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 8):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI()

    def answer_multiple_choice(self, context, question, options) -> str:
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context or not options:
            return ""
        self._ensure_client()
        try:
            r = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": MC_SYSTEM},
                    {"role": "user", "content": format_mc_user(context, question, options)},
                ],
                max_tokens=self.max_tokens, temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  [WARN] OpenAI MC error: {exc}", file=sys.stderr)
            return ""

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
                    {"role": "system", "content": "Answer concisely using only the context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                ],
                max_tokens=64, temperature=0,
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
                     "content": "Summarize this excerpt from a story or article concisely, "
                                "preserving the main events, characters, relationships, "
                                "motivations, and themes."},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.max_tokens, temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return text[:200]


# ---------------------------------------------------------------------------
# Tier registry — same models as the QASPER harness for cross-dataset parity
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