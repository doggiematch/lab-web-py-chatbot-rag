from openai import OpenAI
import chromadb

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documentos")

historial = {}

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-nomic-embed-text-v1.5",
        input=text
    )
    return response.data[0].embedding

def chat(pregunta: str, session_id: str) -> dict:
    embedding_pregunta = get_embedding(pregunta)

    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=3
    )

    fragmentos = resultados["documents"][0]
    fuentes = list(set([m["filename"] for m in resultados["metadatas"][0]]))
    contexto = "\n\n".join(fragmentos)

    if session_id not in historial:
        historial[session_id] = []

    instrucciones = (
        "Responde solo usando la información del contexto proporcionado. "
        "Si la pregunta no puede responderse con el contexto, di exactamente: "
        "'No tengo información sobre eso'. "
        "No inventes datos, fechas, precios ni lugares que no aparezcan en el contexto."
    )

    mensajes = historial[session_id] + [
        {
            "role": "user",
            "content": f"{instrucciones}\n\nContexto:\n{contexto}\n\nPregunta: {pregunta}"
        }
    ]

    respuesta = client.chat.completions.create(
        model="local-model",
        messages=mensajes
    )

    respuesta_texto = respuesta.choices[0].message.content

    historial[session_id].append({"role": "user", "content": pregunta})
    historial[session_id].append({"role": "assistant", "content": respuesta_texto})

    return {
        "respuesta": respuesta_texto,
        "fuentes": fuentes,
        "session_id": session_id,
        "fragmentos_usados": len(fragmentos)
    }