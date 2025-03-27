from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.vectorstores import Chroma
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import GenerativeAgentMemory
from langchain_core.language_models.chat_models import BaseChatModel
import faiss
import math
import os
from dotenv import load_dotenv
from pathlib import Path
import pickle
from typing import Optional, List, Dict, Union
from datetime import datetime
from langchain_core.documents import Document

# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Load the .env file from parent directory
dotenv_path = parent_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


class MemoryDocstore(Docstore, AddableMixin):

    def __init__(self, _dict: Optional[Dict[str, Document]] = None, path: str = None):
        """
        Initialize with dict, and optionally load from a persistent path.
        If a path is provided and the file exists, it loads the docstore from there.
        """
        self.path = path
        if self.path and os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self._dict = pickle.load(f)
        else:
            self._dict = _dict if _dict is not None else {}
            if self.path:
                self.dump()

    def add(self, texts: Dict[str, Document]) -> None:
        """Add texts to in memory dictionary.

        Args:
            texts: dictionary of id -> document.

        Returns:
            None
        """
        overlapping = set(texts).intersection(self._dict)
        if overlapping:
            raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        self._dict = {**self._dict, **texts}
        self.dump()

    def delete(self, ids: List) -> None:
        """Deleting IDs from in memory dictionary."""
        overlapping = set(ids).intersection(self._dict)
        if not overlapping:
            raise ValueError(f"Tried to delete ids that does not  exist: {ids}")
        for _id in ids:
            self._dict.pop(_id)
        self.dump()

    def search(self, search: str) -> Union[str, Document]:
        """Search via direct lookup.

        Args:
            search: id of a document to search for.

        Returns:
            Document if found, else error message.
        """
        if search not in self._dict:
            return f"ID {search} not found."
        else:
            return self._dict[search]

    def dump(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "wb") as f:
                pickle.dump(self._dict, f)

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)

def get_memory_retriever(name_of_memory: str):
    """Create a new or retrieve an existing vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    store_path = f"agent_memories/{name_of_memory}"
    # docstore_path = f"{store_path}/docstore.pkl"

    # # Try to load an existing FAISS store
    # if os.path.exists(store_path):
    #     try:
    #         vectorstore = FAISS.load_local(
    #             store_path,
    #             embeddings_model,
    #             relevance_score_fn=relevance_score_fn,
    #             allow_dangerous_deserialization=True # we trust the files under agent_memories
    #         )
    #         # docstore = load_docstore(docstore_path)
    #         # vectorstore.docstore = docstore
    #     except Exception as e:
    #         print(f"[WARN] Failed to load FAISS store: {e}")
    #         vectorstore = None
    # else:
    #     print(f"[INFO] No FAISS store found at {store_path}")
    #     vectorstore = None

    # If not found or failed, create a new one
    if vectorstore is None:
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        # docstore = InMemoryDocstore({})
        docstore = MemoryDocstore({}, path=f"{store_path}/docstore.pkl")
        vectorstore = FAISS(
            embedding_function=embeddings_model.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id={},
            relevance_score_fn=relevance_score_fn,
        )
        # os.makedirs(store_path, exist_ok=True)
        # vectorstore.save_local(store_path)
        # save_docstore(docstore, docstore_path)

    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

def get_memory(llm: BaseChatModel, name_of_memory: str):
    return GenerativeAgentMemory(
        llm=llm,
        memory_retriever=get_memory_retriever(name_of_memory),
        verbose=False,
        reflection_threshold=8
    )

# def save_memory(memory: GenerativeAgentMemory, name_of_memory: str):
#     store_path = f"agent_memories/{name_of_memory}"
#     docstore_path = f"{store_path}/docstore.pkl"

#     memory.memory_retriever.vectorstore.save_local(store_path)
#     save_docstore(memory.memory_retriever.vectorstore.docstore, docstore_path)