import logging

logging.basicConfig(level=logging.ERROR)
# from datetime import datetime, timedelta
# from typing import List

# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
# from termcolor import colored
# import openai
import os
# from dotenv import load_dotenv
# from pathlib import Path
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    # GenerativeAgentMemory,
)
# import math
# import faiss
from ..memory.memory import get_memory

# print(os.environ['OPENAI_API_KEY'])

def agent_generator(
    name: str, 
    age: int, 
    traits: str, 
    status: str,
    llm: BaseChatModel,
    # memory: GenerativeAgentMemory
) -> GenerativeAgent:
    agent = GenerativeAgent(
        name=name,
        age=age,
        traits=traits,
        status=status,
        # memory_retriever=create_new_memory_retriever(),
        llm=llm,
        memory=get_memory(llm, name + "_memory"),
    )
    return agent