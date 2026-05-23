from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from collections import defaultdict
from datetime import datetime, timedelta
from chatbot import chat, historial
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot RAG ~ Guía de Matera")

solicitudes_por_ip = defaultdict(list)

class PreguntaRequest(BaseModel):
    pregunta: str = Field(max_length=500)
    session_id: str

def check_rate_limit(ip: str):
    ahora = datetime.now()
    minuto_atras = ahora - timedelta(minutes=1)
    solicitudes_por_ip[ip] = [t for t in solicitudes_por_ip[ip] if t > minuto_atras]
    if len(solicitudes_por_ip[ip]) >= 10:
        raise HTTPException(status_code=429, detail="Demasiadas peticiones. Espera un minuto.")
    solicitudes_por_ip[ip].append(ahora)

def contiene_datos_personales(texto: str) -> bool:
    import re
    patron_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    patron_nombre = r'\bmi nombre es\b|\bme llamo\b|\bsoy [A-Z][a-z]+\b'
    return bool(re.search(patron_email, texto) or re.search(patron_nombre, texto, re.IGNORECASE))

@app.post("/chat")
async def endpoint_chat(request: Request, body: PreguntaRequest):
    ip = request.client.host
    check_rate_limit(ip)
    logger.info(f"Peticion de {ip} - session: {body.session_id} - longitud pregunta: {len(body.pregunta)}")
    advertencia = None
    if contiene_datos_personales(body.pregunta):
        advertencia = "Tu pregunta parece contener datos personales. Ten cuidado con la información que compartes."
    resultado = chat(body.pregunta, body.session_id)
    if advertencia:
        resultado["advertencia"] = advertencia
    return resultado

@app.get("/chat/history/{session_id}")
async def obtener_historial(session_id: str):
    if session_id not in historial:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    return {"session_id": session_id, "historial": historial[session_id]}

@app.get("/documentos")
async def listar_documentos():
    import os
    archivos = [f for f in os.listdir("./docs") if f.endswith(".txt")]
    return {"documentos": archivos, "total": len(archivos)}