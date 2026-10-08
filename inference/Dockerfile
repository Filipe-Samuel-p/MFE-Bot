# Worker de inferência (Fase 2 — Online). Constrói o grafo transdutivo,
# carrega os 3 modelos serializados (Fase 1, gerados fora do container) e
# popula o PostgreSQL. Contexto de build: raiz do projeto (ver docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

COPY inference/requirements.txt inference/requirements.txt
RUN pip install --no-cache-dir -r inference/requirements.txt

COPY inference/ inference/

# data/ e inference/models/ NÃO entram na imagem (gitignored, pesados,
# gerados/baixados fora do build) — são montados como volumes no compose.
CMD ["python", "inference/worker/run_inference.py"]
