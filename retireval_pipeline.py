from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

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
query = "When did Google became a public company?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display Retrieved Chunks
print("----- Context -----")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")