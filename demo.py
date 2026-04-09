import os
from pathlib import Path

# ---------------------------------------------------------------------
# 0) Environment setup
# ---------------------------------------------------------------------

# NOTE: An OpenAI API key must be set for initialization if you use the OpenAI-backed
# models in EmbeddingModels.py / SummarizationModels.py / QAModels.py.
# If you are not using those models, a placeholder value is fine.
print(os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------
# 1) Imports
# ---------------------------------------------------------------------

from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import BaseEmbeddingModel, SBertEmbeddingModel
from raptor.QAModels import BaseQAModel
from raptor.SummarizationModels import BaseSummarizationModel


# ---------------------------------------------------------------------
# 2) Load demo text
# ---------------------------------------------------------------------

def load_demo_text() -> str:
    """
    Load text from sample.txt if it exists beside this script.
    Otherwise fall back to a built-in short story.
    """
    script_dir = Path(__file__).resolve().parent
    candidate_paths = [
        script_dir / "sample.txt",
        script_dir / "demo" / "sample.txt",
    ]

    for path in candidate_paths:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            print(f"Loaded demo text from: {path}")
            print(text[:200], "...\n")
            return text

    text = """
    Cinderella lived with her cruel stepmother and two jealous stepsisters, who forced
    her to work day and night. When the royal ball was announced, Cinderella wished to go
    but was forbidden. Her fairy godmother appeared and transformed a pumpkin into a coach,
    mice into horses, and Cinderella's worn clothes into a beautiful gown with glass
    slippers. She attended the ball and danced with the prince, but she had to leave before
    midnight, when the magic would end. As she fled, one glass slipper was left behind.
    The prince searched the kingdom for the woman whose foot fit the slipper. When he came
    to Cinderella's house, the slipper fit her perfectly, and he recognized her as the one
    he loved. Cinderella married the prince and finally found her happy ending.
    """.strip()

    print("Using built-in sample text.")
    print(text[:200], "...\n")
    return text


# ---------------------------------------------------------------------
# 3) Building the tree with the default pipeline
# ---------------------------------------------------------------------

def run_default_demo(text: str):
    print("=" * 80)
    print("DEFAULT DEMO: build -> query -> save -> reload")
    print("=" * 80)

    RA = RetrievalAugmentation()

    print("\nBuilding the tree...")
    RA.add_documents(text)

    print("\nQuerying from the tree...")
    question = "How did Cinderella reach her happy ending?"
    answer = RA.answer_question(question=question)
    print(f"Question: {question}")
    print(f"Answer:   {answer}")

    save_path = Path(__file__).resolve().parent / "demo_tree.pkl"
    print(f"\nSaving tree to: {save_path}")
    RA.save(str(save_path))

    print("Reloading the tree...")
    RA_reloaded = RetrievalAugmentation(tree=str(save_path))
    reloaded_answer = RA_reloaded.answer_question(question=question)
    print(f"Reloaded answer: {reloaded_answer}")

    print("\nRetrieving raw context and layer information...")
    context, layer_info = RA.retrieve(
        question=question,
        top_k=5,
        collapse_tree=True,
        return_layer_information=True,
    )
    print("Retrieved context preview:")
    print(context[:500], "...\n")
    print("Layer info:")
    print(layer_info)


# ---------------------------------------------------------------------
# 4) Example: using custom models
# ---------------------------------------------------------------------

class LocalEchoSummarizationModel(BaseSummarizationModel):
    """
    Tiny local summarizer for demonstration purposes.
    It simply truncates the input to the first few words.
    Replace this with Gemma, Llama, Mistral, T5, etc.
    """
    def summarize(self, context, max_tokens=150):
        words = context.replace("\n", " ").split()
        return " ".join(words[: min(50, len(words))])


class LocalHeuristicQAModel(BaseQAModel):
    """
    Very small QA placeholder. It returns the first sentence in the retrieved context
    that mentions useful Cinderella-related answer cues.
    """
    def answer_question(self, context, question):
        sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
        keywords = ["slipper", "prince", "fairy godmother", "ball", "married", "happy ending"]

        for sentence in sentences:
            lower = sentence.lower()
            if any(k in lower for k in keywords):
                return sentence + "."
        return sentences[0] + "." if sentences else "No answer found."


def run_custom_model_demo(text: str):
    print("=" * 80)
    print("CUSTOM MODEL DEMO")
    print("=" * 80)

    # You can swap in your own summarization / QA / embedding models here.
    # We reuse SBERT embeddings from the uploaded code because they are already supported
    # and easy to replace with another sentence-transformers model if desired.
    config = RetrievalAugmentationConfig(
        summarization_model=LocalEchoSummarizationModel(),
        qa_model=LocalHeuristicQAModel(),
        embedding_model=SBertEmbeddingModel(
            model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"
        ),
    )

    RA = RetrievalAugmentation(config=config)

    print("\nBuilding the tree with custom local summarization / QA models...")
    RA.add_documents(text)

    question = "How did Cinderella reach her happy ending?"
    answer = RA.answer_question(question=question)
    print(f"Question: {question}")
    print(f"Answer:   {answer}")


# ---------------------------------------------------------------------
# 5) Main entry point
# ---------------------------------------------------------------------

def main():
    text = load_demo_text()

    # This mirrors the notebook's main flow.
    run_default_demo(text)

    # This mirrors the notebook's "use other models" section, but with lightweight
    # local classes you can run or replace more easily.
    try:
        run_custom_model_demo(text)
    except Exception as exc:
        print("\nCustom model demo could not be completed.")
        print(f"Reason: {exc}")


if __name__ == "__main__":
    main()

