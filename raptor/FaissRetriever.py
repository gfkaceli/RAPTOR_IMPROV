import random
from concurrent.futures import ProcessPoolExecutor

import faiss
import numpy as np
import tiktoken
from tqdm import tqdm

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
#from .Retrievers import BaseRetriever
#from .utils import split_text

class FaissRetrieverConfig:
    def __init__(self, max_tokens=100,
                 max_context=3500,
                 use_top_k=False,
                 embedding_model=None,
                 top_k=5,
                 tokenizer=tiktoken.get_encoding("cl100k_base"),
                 embedding_model_string=None
                 ):
        pass
    def log_config(self):
        pass


