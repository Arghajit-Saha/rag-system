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

chat_history = []

def ask_question(query):
    if chat_history:
        messages = (
            [{"role": "system", "content": "Given the chat history, rewrite the new question to be standalone and searchable. Strictly return the rewritten question."}]
            + chat_history
            + [{"role": "user", "content": query}]
        )

        with OpenRouter(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ) as open_router:
            res = open_router.chat.send(
                messages=messages, 
                model="openai/gpt-4o-mini", 
                stream=False
            )
        question = res.choices[0].message.content.strip()
        print(f"Searching for: {question}!")
    else:
        question = query

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.3
        }
    )


    relevant_docs = retriever.invoke(question)

    combined_input = f"""Based on the following documents, please answer this question: {question}

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
    
    answer = res.choices[0].message.content.strip()
    print(f"\nAnswer:\n{answer}")

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})


def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()