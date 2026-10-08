# MFE-Bot 🤖

Detector de bots em redes sociais baseado em **Stacking Ensemble com Early Exit**, combinando um modelo especialista em metadados de perfil (Random Forest) com um modelo especialista em topologia de rede (BotRGCN).

---

## 🌐 Idiomas / Languages
- [Português](#-português) 🇧🇷
- [English](#-english) 🇺🇸

---

## 🇧🇷 Português

### 📝 Visão Geral
O **MFE-Bot** (*Multi-Feature Ensemble Bot Detector*) utiliza dois classificadores heterogêneos cujos vereditos são combinados por uma **regra semântica de convergência**:

*   **Random Forest:** Analisa features tabulares do perfil (seguidores, amigos, verificado, etc.).
*   **BotRGCN:** Analisa o grafo heterogêneo de seguidores/seguidos com PyTorch Geometric.
*   **Regressão Logística (Meta-classificador):** Acionada apenas quando os dois modelos base discordam sobre a classe.

> 🎓 **Trabalho de Conclusão de Curso** — Ciência da Computação.

### ⚙️ Lógica de Roteamento (Early Exit)
A divergência é detectada puramente por discordância de classe, otimizando o processo de decisão:

```python
pred_rf  = p_rf  >= 0.5
pred_gnn = p_gnn >= 0.5

if pred_rf == pred_gnn:
    # CONVERGÊNCIA — mesma classe, veredito = média aritmética
    flag_divergente = False
    veredito_final  = (p_rf + p_gnn) / 2
else:
    # DIVERGÊNCIA — classes opostas, chama a Regressão Logística
    flag_divergente = True
    veredito_final  = meta_classificador.predict_proba([[p_rf, p_gnn]])[0][1]
```

### 📂 Estrutura do Projeto
```text
mfe-bot/
├── docker-compose.yml
├── init.sql
├── data/                       # Dataset TwiBot-20 (não versionado)
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── inference/                  # Serviço de Inferência (Python)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/                 # Modelos serializados (não versionado)
│   ├── training/               # Fase 1 (offline) — Treinamento
│   ├── worker/                 # Fase 2 (online) — Execução
│   └── evaluation/             # Métricas e Avaliação
└── api/                        # API de Orquestração (Golang)
    └── internal/
        ├── handler/
        ├── service/
        ├── repository/
        └── model/
```

### 📊 Dataset — TwiBot-20
Arquivos localizados no diretório `data/`:

| Arquivo | Tamanho | Uso |
| :--- | :--- | :--- |
| `train.json` | 239 MB | Treino dos modelos base (RF e GNN) |
| `dev.json` | 68,9 MB | Treino da Regressão Logística |
| `test.json` | 34 MB | Inferência online (Fase 2) |

*   `label: "0"` = Humano | `label: "1"` = Bot
*   Usuários sem vizinhos (`neighbor: null`) são processados apenas pelo Random Forest.

### 🗄️ Schema do Banco de Dados
```sql
CREATE TABLE deteccao_bots (
    id               SERIAL PRIMARY KEY,
    user_id          VARCHAR(50)   NOT NULL,
    screen_name      VARCHAR(100),
    p_rf             FLOAT         NOT NULL,
    p_gnn            FLOAT         NOT NULL,
    p_meta           FLOAT,
    flag_divergente  BOOLEAN       NOT NULL,
    veredito_final   FLOAT         NOT NULL,
    label_real       INTEGER,
    data_analise     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
```

### 🚀 Fluxo de Execução

#### Fase 1 — Offline (Execução Local)
1.  **Treinar Random Forest:** `python inference/training/train_rf.py`
2.  **Treinar BotRGCN:** `python inference/training/train_gnn.py`
3.  **Treinar Meta-classificador:** `python inference/training/train_lr.py`

#### Fase 2 — Online (Docker)
```bash
# Sobe o Postgres, roda o worker (processa test.json e popula o banco) e
# inicia a API. O worker é um job: processa e encerra; a API permanece de pé.
docker compose up --build
# Consultar métricas
curl http://localhost:8080/v1/metrics/summary
```

---

## 🇺🇸 English

### 📝 Overview
**MFE-Bot** (*Multi-Feature Ensemble Bot Detector*) is a social-network bot detector based on **Stacking Ensemble with Early Exit**. It combines two heterogeneous specialist classifiers:

*   **Random Forest:** Analyzes tabular profile metadata (followers, friends, verified, etc.).
*   **BotRGCN:** Analyzes the heterogeneous follower/following graph via PyTorch Geometric.
*   **Logistic Regression (Meta-classifier):** Invoked only when the two base models disagree on the predicted class.

> 🎓 **Computer Science Thesis Project**.

### ⚙️ Routing Logic (Early Exit)
Divergence is detected purely by class disagreement, ensuring efficient decision-making:

```python
pred_rf  = p_rf  >= 0.5
pred_gnn = p_gnn >= 0.5

if pred_rf == pred_gnn:
    # CONVERGENCE — same class, verdict = arithmetic mean
    flag_divergent = False
    final_verdict  = (p_rf + p_gnn) / 2
else:
    # DIVERGENCE — opposite classes, escalate to Logistic Regression
    flag_divergent = True
    final_verdict  = meta_classifier.predict_proba([[p_rf, p_gnn]])[0][1]
```

### 📂 Project Structure
```text
mfe-bot/
├── docker-compose.yml
├── init.sql
├── data/                       # TwiBot-20 Dataset (not versioned)
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── inference/                  # Inference Service (Python)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/                 # Serialized models (not versioned)
│   ├── training/               # Phase 1 (offline) — Training
│   ├── worker/                 # Phase 2 (online) — Execution
│   └── evaluation/             # Metrics & Evaluation
└── api/                        # Orchestration API (Golang)
    └── internal/
        ├── handler/
        ├── service/
        ├── repository/
        └── model/
```

### 📊 Dataset — TwiBot-20
Files located in the `data/` directory:

| File | Size | Purpose |
| :--- | :--- | :--- |
| `train.json` | 239 MB | Train base models (RF and GNN) |
| `dev.json` | 68.9 MB | Train the Logistic Regression meta-classifier |
| `test.json` | 34 MB | Online inference (Phase 2) |

*   `label: "0"` = Human | `label: "1"` = Bot
*   Users with no neighbors (`neighbor: null`) are fed only to the Random Forest.

### 🗄️ Database Schema
```sql
CREATE TABLE deteccao_bots (
    id               SERIAL PRIMARY KEY,
    user_id          VARCHAR(50)   NOT NULL,
    screen_name      VARCHAR(100),
    p_rf             FLOAT         NOT NULL,
    p_gnn            FLOAT         NOT NULL,
    p_meta           FLOAT,
    flag_divergente  BOOLEAN       NOT NULL,
    veredito_final   FLOAT         NOT NULL,
    label_real       INTEGER,
    data_analise     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
```

### 🚀 Execution Flow

#### Phase 1 — Offline (Local Run)
1.  **Train Random Forest:** `python inference/training/train_rf.py`
2.  **Train BotRGCN:** `python inference/training/train_gnn.py`
3.  **Train Meta-classifier:** `python inference/training/train_lr.py`

#### Phase 2 — Online (Docker)
```bash
# Starts Postgres, runs the worker (processes test.json and populates the
# database) and starts the API. The worker is a one-off job: it processes
# and exits; the API stays up.
docker compose up --build
# Query metrics
curl http://localhost:8080/v1/metrics/summary
```

### 🛠️ Stack
| Component | Technology |
| :--- | :--- |
| **Inference Worker** | Python 3.11 |
| **Random Forest** | scikit-learn |
| **BotRGCN** | PyTorch + PyTorch Geometric |
| **Meta-classifier** | scikit-learn (LogisticRegression) |
| **API** | Golang 1.22 |
| **Database** | PostgreSQL 16 |
| **Orchestration** | Docker Compose |