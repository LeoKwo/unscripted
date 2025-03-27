from agents.character.agent import agent_generator
from langchain_ollama import ChatOllama
import json

LLM = ChatOllama(model="llama3.1")

tommy = agent_generator("Tommy", 25, "kind, helpful, friendly", "looking for a job", LLM)

print(tommy.generate_reaction("Tommy was rejected by his dream company and is feeling down."))
# print(tommy.generate_reaction("Tommy won a lottery and is feeling happy."))

print("\nMemories\n")
# print(tommy.memory.memory_retriever.vectorstore.get())
print([doc.model_dump() for doc in tommy.memory.memory_retriever.vectorstore.docstore._dict.values()])