"""
Conexão com o PostgreSQL (fonte única da verdade do MFE-Bot).

Compartilhado entre o worker (`worker/run_inference.py`) e a avaliação
(`evaluation/metrics.py`). Lê credenciais de variáveis de ambiente, com
defaults para rodar o worker LOCAL contra o Postgres do docker-compose
(`localhost:5432`, db/usuário/senha `mfebot`).
"""

import os

import psycopg2


def connect():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ.get("DB_NAME", "mfebot"),
        user=os.environ.get("DB_USER", "mfebot"),
        password=os.environ.get("DB_PASSWORD", "mfebot"),
    )
