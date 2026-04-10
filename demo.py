from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Some project modules expect this variable to exist, even though we do not use it here.
os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")

from raptor.EmbeddingModels import BaseEmbeddingModel, SBertEmbeddingModel
from raptor.QAModels import BaseQAModel
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.SummarizationModels import BaseSummarizationModel


# ---------------------------------------------------------------------
# 1) Load demo text
# ---------------------------------------------------------------------

def load_demo_text() -> str:
    """Load sample text from disk or use a built-in example."""
    script_dir = Path(__file__).resolve().parent
    candidate_paths = [
        script_dir / "sample.txt",
        script_dir / "demo" / "sample.txt",
    ]

    for path in candidate_paths:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            print(f"Loaded demo text from: {path}")
            print(text[:300], "...\n")
            return text

    text = (
        "Cinderella lived with her cruel stepmother and two jealous stepsisters, who forced "
        "her to work day and night. When the royal ball was announced, Cinderella wished to go "
        "but was forbidden. Her fairy godmother appeared and transformed a pumpkin into a coach, "
        "mice into horses, and Cinderella's worn clothes into a beautiful gown with glass slippers. "
        "She attended the ball and danced with the prince, but she had to leave before midnight, "
        "when the magic would end. As she fled, one glass slipper was left behind. The prince "
        "searched the kingdom for the woman whose foot fit the slipper. When he came to "
        "Cinderella's house, the slipper fit her perfectly, and he recognized her as the one "
        "he loved. Cinderella married the prince and finally found her happy ending.\n\n"
        "The story is often used to illustrate transformation, perseverance, and recognition. "
        "In many retellings, the fairy godmother symbolizes hope, while the lost slipper becomes "
        "the key piece of evidence that allows the prince to identify Cinderella."
    )
    print("Using built-in sample text.")
    print(text[:300], "...\n")
    return text


# ---------------------------------------------------------------------
# 2) Open-source local model wrappers
# ---------------------------------------------------------------------

class LocalBartSummarizationModel(BaseSummarizationModel):
    """Open-source summarizer with a safe heuristic fallback."""

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._load_error: Optional[Exception] = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            self._load_error = exc

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text:
            return ""

        self._ensure_loaded()
        if self._pipeline is not None:
            try:
                result = self._pipeline(
                    text,
                    max_new_tokens=min(int(max_tokens), 128),
                    min_new_tokens=20,
                    do_sample=False,
                    truncation=True,
                )
                return result[0]["summary_text"].strip()
            except Exception:
                pass

        # Fallback: simple extractive summary
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(sentences[:2]) + ("." if sentences else "")


class LocalFlanQAModel(BaseQAModel):
    """Open-source QA model with a heuristic fallback."""

    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._load_error: Optional[Exception] = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            self._load_error = exc

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return "No context available."

        self._ensure_loaded()
        if self._pipeline is not None:
            prompt = (
                "Answer the question using the provided context. "
                "If the answer is not in the context, say you do not know.\n\n"
                f"Context: {context}\n\nQuestion: {question}"
            )
            try:
                result = self._pipeline(prompt, max_new_tokens=64, do_sample=False)
                return result[0]["generated_text"].strip()
            except Exception:
                pass

        # Fallback: heuristic sentence extraction
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        question_terms = {
            token.lower().strip("?,.!:;\"'()")
            for token in question.split()
            if len(token.strip("?,.!:;\"'()")) > 2
        }
        best_sentence = ""
        best_score = -1
        for sentence in sentences:
            lowered = sentence.lower()
            score = sum(term in lowered for term in question_terms)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence + "." if best_sentence else "No answer found in context."


class TinyExtractiveSummarizationModel(BaseSummarizationModel):
    """Very lightweight open-source-friendly fallback demo summarizer."""

    def summarize(self, context, max_tokens=150):
        words = " ".join(str(context).split()).split()
        return " ".join(words[: min(60, len(words))])


class TinyHeuristicQAModel(BaseQAModel):
    """Very lightweight open-source-friendly fallback demo QA model."""

    def answer_question(self, context, question):
        sentences = [s.strip() for s in str(context).replace("\n", " ").split(".") if s.strip()]
        keywords = [token.lower() for token in str(question).split() if len(token) > 3]
        best = ""
        best_score = -1
        for sentence in sentences:
            score = sum(keyword in sentence.lower() for keyword in keywords)
            if score > best_score:
                best = sentence
                best_score = score
        return best + "." if best else "No answer found."


# ---------------------------------------------------------------------
# 3) Config helpers
# ---------------------------------------------------------------------

def make_open_source_config() -> RetrievalAugmentationConfig:
    """Primary local config using free/open-source models."""
    return RetrievalAugmentationConfig(
        embedding_model=SBertEmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        summarization_model=LocalBartSummarizationModel(
            model_name="sshleifer/distilbart-cnn-12-6"
        ),
        qa_model=LocalFlanQAModel(model_name="google/flan-t5-base"),
        tb_max_tokens=80,
        tb_num_layers=3,
        tb_summarization_length=80,
        tr_top_k=5,
        tr_selection_mode="top_k",
    )


def make_lightweight_config() -> RetrievalAugmentationConfig:
    """Fallback config with SBERT + tiny local logic only."""
    return RetrievalAugmentationConfig(
        embedding_model=SBertEmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        summarization_model=TinyExtractiveSummarizationModel(),
        qa_model=TinyHeuristicQAModel(),
        tb_max_tokens=80,
        tb_num_layers=3,
        tb_summarization_length=80,
        tr_top_k=5,
        tr_selection_mode="top_k",
    )


# ---------------------------------------------------------------------
# 4) Demo flow
# ---------------------------------------------------------------------

def build_query_save_reload(text: str, config: RetrievalAugmentationConfig, label: str) -> None:
    print("=" * 90)
    print(label)
    print("=" * 90)

    ra = RetrievalAugmentation(config=config)

    print("\nBuilding the tree...")
    ra.add_documents(text)

    question = "How did Cinderella reach her happy ending?"
    print(f"\nQuestion: {question}")

    context, layer_info = ra.retrieve(
        question=question,
        top_k=5,
        collapse_tree=True,
        return_layer_information=True,
    )
    print("\nRetrieved context preview:")
    print(context[:600], "...\n")
    print("Layer information:")
    print(layer_info)

    answer = ra.answer_question(question=question)
    print("\nAnswer:")
    print(answer)

    save_path = Path(__file__).resolve().parent / "demo_tree.pkl"
    print(f"\nSaving tree to: {save_path}")
    ra.save(str(save_path))

    print("Reloading the tree and asking again...")
    reloaded_ra = RetrievalAugmentation(config=config, tree=str(save_path))
    reloaded_answer = reloaded_ra.answer_question(question=question)
    print("Reloaded answer:")
    print(reloaded_answer)


# ---------------------------------------------------------------------
# 5) Main
# ---------------------------------------------------------------------

def main() -> None:
    text = load_demo_text()

    try:
        build_query_save_reload(
            text,
            make_open_source_config(),
            "OPEN-SOURCE DEMO: SBERT + DistilBART + FLAN-T5",
        )
    except Exception as exc:
        print("\nPrimary open-source demo could not complete.")
        print(f"Reason: {exc}")
        print("\nFalling back to a lighter fully local demo...\n")
        build_query_save_reload(
            text,
            make_lightweight_config(),
            "LIGHTWEIGHT LOCAL DEMO: SBERT + heuristic summarization/QA",
        )


if __name__ == "__main__":
    main()
