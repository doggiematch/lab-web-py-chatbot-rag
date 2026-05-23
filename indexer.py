import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import tiktoken

load_dotenv()

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documentos")

def chunk_text(text, max_tokens=300):
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_tokens = len(enc.encode(word))
        if current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-nomic-embed-text-v1.5",
        input=text
    )
    return response.data[0].embedding

def index_documents():
    docs_path = "./docs"
    total_chunks = 0
    total_tokens = 0
    enc = tiktoken.get_encoding("cl100k_base")

    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = chunk_text(content)

            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                tokens = len(enc.encode(chunk))
                total_tokens += tokens

                collection.upsert(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"{filename}_chunk_{i}"],
                    metadatas=[{"filename": filename, "chunk_id": i}]
                )
                total_chunks += 1

            print(f" {filename}: {len(chunks)} chunks indexados")

    print(f"\n Resumen:")
    print(f"   Documentos procesados: {len(os.listdir(docs_path))}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total tokens procesados: {total_tokens}")

if __name__ == "__main__":
    index_documents()