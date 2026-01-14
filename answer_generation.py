import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from openrouter import OpenRouter

load_dotenv()

persistent_directory = "db/chroma_db"

# Loading Text Embedding Model
embedding_model = OllamaEmbeddings(model='nomic-embed-text:v1.5')

# Loading Vector Store
db = Chroma(
    persist_directory=persistent_directory, 
    embedding_function=embedding_model, 
    collection_metadata={"hnsw:space": "cosine"}
)

# User Query 
query = input("Enter Your Question: ")

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3
    }
)

relevant_docs = retriever.invoke(query)

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from this documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": combined_input}
]

with OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
) as open_router:
    res = open_router.chat.send(
        messages=messages, 
        model="openai/gpt-4o-mini", 
        stream=False
    )

print("\n------- Generated Response --------")
print("Content Only: ")
print({res.choices[0].message.content})