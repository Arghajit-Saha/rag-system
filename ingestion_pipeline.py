import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists. Please create it and add your files.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your documents.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocuments {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content Length: {len(doc.page_content)} characters")
        print(f"  Content Preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting documnets into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and and stpring in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

    print("----- Creating Vector Store -----")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space" : "cosine"}
    )
    print("----- Finished creating vector store -----")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Hello World")

    #1. Loading the Documents
    documents = load_documents(docs_path="docs")

    #2. Chunking the Documents
    chunks = split_documents(documents)

    #3. Embedding and storing in Vector DB
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()