# Registro da Sessão — MFE-Bot (Fases 1, 2 e 3)

> Documento de continuidade. Resume tudo que foi feito nesta sessão para que um
> novo contexto recupere o estado. **Data:** 2026-06-29 → 2026-07-03. Não contém código —
> aponta arquivos, funções, decisões e resultados.

---

## 0. Resumo executivo

**Fase 1 (Ciência de Dados) CONCLUÍDA** — os 3 modelos estão treinados e
serializados:
- ✅ Schema do banco (`init.sql`) corrigido para bater com o CLAUDE.md.
- ✅ **Random Forest calibrada** (F1 holdout 0.8475).
- ✅ Embeddings **RoBERTa** pré-computados (`text_emb.npz`, ~71,5 min).
- ✅ **BotRGCN multimodal** (RoBERTa + perfil + rede) — VAL F1 **0.8773**.
- ✅ **Regressão Logística** — com o multimodal, o coef do GNN foi de ~0 para
  **+2,92** e a divergência subiu de 4,1% → **14,2%** (heterogeneidade validada).
- ✅ `CLAUDE.md` atualizado para refletir o TCC (GNN multimodal, 3 fases, etc.).

**Fase 3 (XAI — Explicabilidade Post-hoc) CONCLUÍDA** — módulos sob demanda implementados e testados:
- ✅ `inference/xai/explain_rf.py` — SHAP `TreeExplainer` sobre o RF calibrado (lazy eval + CLI).
- ✅ `inference/xai/explain_gnn.py` — GNNExplainer (rede) + Gradient×Input (texto) (lazy eval + CLI).
- ✅ `inference/xai/sample_selector.py` — amostra estratificada em 4 categorias (CLI `--limit N`).
- ✅ Explicações de rede (`network_explanation`) e de perfil (`profile_feature_importance`) testadas ao vivo com usuário real.

**Fase 2 (Engenharia / Online) CONCLUÍDA** — banco populado e tese sustentada:
- ✅ `inference/worker/router.py` — Roteamento por Cruzamento de Fronteira.
- ✅ `inference/worker/run_inference.py` — orquestrador (grafo transdutivo + streaming screen_names).
- ✅ `inference/evaluation/metrics.py` — tabela de métricas do TCC.
- ✅ Banco populado: **1.183 usuários** de test.json gravados.
- ✅ **TESE SUSTENTADA:** F1(MFE-Bot)=**0.8782** > RF=0.8507 e BotRGCN=0.8751.
- ⏭️ **Próximo:** API Golang (Fase 2B) e XAI (Fase 3).

---

## 1. Ambiente

- Python `3.13` em `.venv/` (criado nesta sessão; `.venv/` no `.gitignore`).
- Pacotes instalados: `numpy pandas scikit-learn joblib ijson torch
  torch-geometric transformers`. (sklearn 1.9.0, torch 2.12.1, pyg 2.8.0)
- **MPS (GPU Metal) disponível** no Mac → usado no RoBERTa.
- Dados em `data/`: `train.json` (8.278 usuários), `dev.json` (2.365),
  `test.json` (1.183). **`support.json` NÃO existe** (faltou no dataset).

---

## 2. Particularidades do TwiBot-20 descobertas (importante)

- O **`ID` de topo é limpo** (`"17461978"`); já `profile.id` tem espaço no fim
  (`"17461978 "`). → usar sempre o `ID` de topo como chave de nó.
- **Todos os valores de `profile` são strings com espaço no fim**; booleanos vêm
  como texto (`"True "`, `"False "`, `"None "`). Tratado em `features.py`.
- `label` é string `"0"`/`"1"`. `neighbor` pode ser `null`. `tweet` pode ser `null`.
- **Arestas:** só ~8,1% das arestas citadas conectam dois usuários do dataset
  (o resto aponta pro `support set` ausente). Por isso o grafo é construído
  **unindo os 3 splits** (transdutivo) — aí 92,2% dos nós ficam conectados
  (grau médio ~3,1; 923 nós isolados).
- Total de tweets: **~2,0 milhões** (11.746 usuários com tweets, média 170 cada).

---

## 3. Arquivos criados/modificados

### Módulos compartilhados (treino + worker) — em `inference/`
- **`features.py`** — extração das 18 features de perfil.
  - `extract_features(user)` → lista de 18 floats (12 numéricas + 6 categóricas).
  - `parse_label(user)` → int 0/1 ou None.
  - Constantes: `FEATURE_NAMES`, `NUMERIC_FEATURE_NAMES`,
    `CATEGORICAL_FEATURE_NAMES`, `N_NUMERIC` (=12).
  - Helpers de limpeza: `_clean`, `_to_float`, `_to_bool`, `_account_age_days`
    (data ref. fixa 2020-09-28), `_str_len`, `_num_digits`.
- **`graph.py`** — grafo transdutivo.
  - `build_graph(data_dir, include_support=True)` → dict com `ids`, `id_to_idx`,
    `X_raw` [N,18], `des_emb`/`tweet_emb` [N,768] (ou None se cache ausente),
    `y`, `split` (0=train 1=dev 2=test 3=support), `edge_index`, `edge_type`,
    `has_neighbor`, `feature_names`.
  - Arestas **DIRECIONADAS**: `REL_FOLLOWS=0` (following), `REL_FOLLOWED_BY=1`
    (follower). `NUM_RELATIONS=2`.
  - `_load_text_embeddings(ids)` — lê `models/text_emb.npz` e realinha à ordem dos nós.
- **`botrgcn.py`** — arquitetura.
  - Classe `BotRGCN(des_dim, tweet_dim, num_dim, cat_dim, hidden_dim,
    num_relations, dropout)`: 4 encoders lineares (des/tweet/num/cat → D/4) →
    concat → `lin_in` → `RGCNConv` ×2 → `lin_out` (2 classes). `hidden_dim=128`.
  - `normalize(X_raw, mean, std)` — log1p + z-score (só nas numéricas).
- **`scoring.py`** — scoring dos modelos-base.
  - `load_rf(path)`, `rf_scores(rf, X_raw)` → p_rf.
  - `load_gnn(path)` → (model, bundle); `gnn_scores(model, bundle, graph)` → p_gnn
    (passada única no grafo transdutivo).

### Scripts de treino — em `inference/training/`
- **`encode_text.py`** — pré-computa embeddings RoBERTa (`roberta-base`, MPS,
  mean-pooling mascarado). Por usuário: 1 embedding de `description` + média dos
  embeddings dos tweets. Salva `models/text_emb.npz` (`ids`, `des_emb`, `tweet_emb`).
  Funções: `get_device`, `encode_batch`, `encode_many`, `main`.
- **`train_rf.py`** — RF + `CalibratedClassifierCV(method="isotonic", cv=5)`.
  `n_estimators=300`, `class_weight="balanced"`. Avalia em **holdout interno
  20%** do train (dev intocado), depois retreina em 100% e salva `modelo_rf.pkl`.
- **`train_gnn.py`** — treina BotRGCN multimodal. Normaliza numéricas no train,
  holdout interno 15% para early-stopping (patience 30), pesos de classe.
  Salva bundle `modelo_gnn.pt` (state_dict + dims + `norm_mean`/`norm_std`).
- **`train_lr.py`** — pontua dev com RF+GNN, filtra **divergentes** entre quem
  entra no stacking (`neighbor != null`), treina `LogisticRegression` neles.
  Reporta contagens, F1 por CV e coeficientes (caixa-branca). Salva `modelo_lr.pkl`.

### Outros
- **`init.sql`** — corrigido: tabela `auditoria_deteccao_bots` com todas as
  colunas (incl. `classificacao_bot`, `explicacao_rf/gnn` JSONB) + 4 índices.
  (View removida a pedido do usuário.)
- **`.gitignore`** — adicionado `.venv/`. (`inference/models/` já estava lá.)
- **`CLAUDE.md`** — atualizado para o TCC (BotRGCN multimodal, arestas
  direcionadas, 3 fases, RoBERTa na stack, regras 12-15).
- **`code`** (arquivo na raiz) — é uma **cópia idêntica antiga do CLAUDE.md**;
  pode ser deletado (pendente confirmação do usuário).

---

## 4. Resultados dos modelos

### Random Forest (calibrada) — `modelo_rf.pkl` (164 MB)
Holdout interno 20% do train (dev intocado):
| Métrica | TREINO 80% | HOLDOUT 20% |
|---|---|---|
| Acurácia | 0.9850 | 0.8013 |
| Precisão | 0.9741 | 0.7443 |
| Revocação | 1.0000 | 0.9839 |
| **F1** | 0.9869 | **0.8475** |
| Brier | 0.0429 | 0.1395 |

### Embeddings RoBERTa — `text_emb.npz` (61 MB)
- `encode_text.py`, `roberta-base`, device **MPS**, mean-pooling mascarado.
- Por usuário: 1 embedding de `description` (768) + média dos embeddings dos
  tweets (768). Total: 11.826 × 768 (des) + 11.826 × 768 (tweet).
- **Tweets codificados: 1.999.788** (~2,0 M).
- **Config final que funcionou:** padding fixo (`max_length`), batching global
  com `index_add_` no device, **lote 512, max_len 48, autocast fp16**.
- **Tempo:** ~**71,5 min** (vazão estável ~950 tw/s; caiu para ~590 tw/s no fim
  por throttling térmico). 1ª versão (padding dinâmico + lotes por usuário +
  fp32) foi abortada após ~2h quase ociosa — recompilação de kernel do MPS a
  cada formato de tensor era o gargalo.

### BotRGCN — duas versões (a multimodal é a final)

**v1 — SÓ perfil (substituída).** 18 features de perfil como features de nó,
arestas bidirecionais. VAL holdout **F1 0.8487** (recall ~0.99, prec ~0.74).
Problema: idêntica em features ao RF → LR ignorava o GNN (ver abaixo).

**v2 — MULTIMODAL (final) — `modelo_gnn.pt`.** 4 modalidades (RoBERTa
description + tweets + num_prop + cat_prop), arestas direcionadas. Grafo:
11.826 nós, **16.908 arestas direcionadas**, 923 isolados. `hidden_dim=128`,
dropout 0.3, Adam lr 0.01, early stop época 65. Treino rápido (CPU, < 1 min).
| Métrica | TREINO interno | VAL interno (holdout) |
|---|---|---|
| Acurácia | 0.8761 | 0.8598 |
| Precisão | 0.8561 | 0.8260 |
| Revocação | 0.9387 | 0.9353 |
| **F1** | 0.8955 | **0.8773** |

→ **+0.029 de F1 sobre a v1**, e precisão 0.74 → 0.83 (mais equilibrado).

### Regressão Logística — `modelo_lr.pkl` — ANTES vs. DEPOIS do multimodal

Resultado central da tese (heterogeneidade). dev: 2.141 entram no stacking.
| | LR com GNN v1 (só-perfil) | **LR com GNN v2 (multimodal)** |
|---|---|---|
| Divergentes | 88 (4,1%) — 39 bot / 49 humano | **304 (14,2%)** — 87 bot / 217 humano |
| coef `p_rf` | +1.0833 | +0.6449 |
| coef `p_gnn` | **−0.0050** (ignorado) | **+2.9198** (domina) |
| intercepto | −0.5035 | −1.5651 |
| LR F1 (CV 5-fold) | 0.5316 | 0.5075 |
| baseline "sempre RF" | 0.4483 | 0.3934 |
| baseline "sempre GNN" | 0.4815 | 0.2735 |

**Achado-chave (para o TCC):** com o GNN multimodal, (1) a **divergência triplica**
(4,1% → 14,2%) porque os especialistas passam a ver sinais distintos; (2) o
**coeficiente do GNN inverte de ~0 para +2,92**, tornando-se o árbitro dominante;
(3) o LR supera os dois baselines com **margem muito maior** no regime divergente.
Isso valida empiricamente a hipótese de "especialistas heterogêneos cooperando".

---

## 5. Decisões metodológicas tomadas

1. **dev.json intocado** no treino de RF/GNN — usa holdout interno do train.
   (O usuário levantou isso; corrigido.)
2. **LR treinada só nos divergentes** (regime onde de fato atua; coeficientes
   interpretáveis como peso histórico na discordância).
3. **Grafo transdutivo único** (train+dev+test) — necessário pois por-split o
   grafo é inútil (0,6% de arestas internas no test). Padrão do BotRGCN/TwiBot-20.
4. **GNN refatorado para MULTIMODAL** (RoBERTa) — após o usuário enviar o TCC
   (Cap. 3) confirmando que o BotRGCN funde perfil + semântica de tweets. Escolha
   do usuário: **RoBERTa completo, fiel ao paper** (Feng et al., 2021a).
5. **Arestas direcionadas** (preserva direção, TCC §3.3.1) — eram bidirecionais.
6. **Worker NÃO pontua GNN por usuário em streaming** (impossível em grafo
   transdutivo) — faz passada única no grafo e lê p_gnn. Contradiz a letra do
   CLAUDE.md antigo; ajustado no CLAUDE.md (nota transdutiva).

---

## 6. Estado ao fim da sessão / próximos passos

### Fase 1 — concluída ✅
Encoding RoBERTa, GNN multimodal e LR já rodaram (resultados na §4). Os 4
artefatos estão em `inference/models/`: `modelo_rf.pkl` (164 MB),
`modelo_gnn.pt`, `modelo_lr.pkl`, `text_emb.npz` (61 MB).

### Fase 2 — concluída ✅
Router + Worker + Metrics implementados. Banco populado (1.183 linhas).

**Distribuição do roteamento (test.json):**
| Categoria | Quantidade | % |
|---|---|---|
| Convergentes | 913 | 77,2% |
| Divergentes | 161 | 13,6% |
| RF-only (neighbor:null) | 109 | 9,2% |

**Métricas finais no test.json (prova da tese — §3.5.1):**
| Modelo | Acurácia | Precisão | Revocação | F1 |
|---|---|---|---|---|
| RF (isolado) | 0.8157 | 0.7573 | 0.9703 | 0.8507 |
| BotRGCN (isolado) | 0.8588 | 0.8393 | 0.9141 | 0.8751 |
| **MFE-Bot** | **0.8588** | 0.8235 | **0.9406** | **0.8782** |

Interpretação: o ensemble eleva a revocação (detecta mais bots reais) sem custo
significativo em precisão. F1 supera ambos os especialistas — **tese sustentada**.

### Fase 3 — concluída ✅

**Arquivos criados em `inference/xai/`:**

| Arquivo | Responsabilidade |
|---|---|
| `explain_rf.py` | SHAP `TreeExplainer` no RF base (fold 0 do `CalibratedClassifierCV`) |
| `explain_gnn.py` | GNNExplainer (edge_mask → top-k vizinhos) + Gradient×Input (texto) |
| `sample_selector.py` | Seleção de amostra estratificada em 4 categorias via SQL |

**Design de separação de responsabilidades XAI:**
- `explain_rf.py` responde: *"quais features de perfil empurraram a decisão?"* (SHAP por feature)
- `explain_gnn.py` responde: *"quais conexões de rede pesaram e os vizinhos são bots?"* (edge mask → user_ids com `neighbor_label`)

**Lazy evaluation** (ambos os módulos):
1. Verifica `explicacao_rf`/`explicacao_gnn` no banco.
2. Se existir → retorna do cache sem recalcular.
3. Se não existir → calcula, salva com `UPDATE` e retorna.

**GNNExplainer — escolha de design:**
- `_ProfileWrapper` expõe apenas as 18 features de perfil ao explainer (embeddings de texto e `edge_type` como buffers fixos).
- A saída principal é `network_explanation.top_neighbors`: user_ids dos vizinhos mais importantes com `edge_importance` e `neighbor_label` (bot/humano).
- `bot_neighbors_ratio` na `edge_summary` é o indicador de homofilia — assinatura de "rede artificial" do TCC.
- Fallback automático para Gradient×Input se `torch_geometric.explain` não disponível (PyG < 2.3).

**Comandos CLI:**
```bash
.venv/bin/python inference/xai/sample_selector.py --limit 5
.venv/bin/python inference/xai/explain_rf.py <user_id>
.venv/bin/python inference/xai/explain_gnn.py <user_id> [--top-k 10] [--epochs 200]
```

**Endpoint Go** (`GET /v1/audit/explanation/{id}`) — já implementado na sessão anterior.
Retorna os campos JSONB `explicacao_rf` e `explicacao_gnn` (null até os scripts serem rodados).

### Passos seguintes do projeto
- Dockerfiles de worker e api (para rodar tudo via `docker compose up`).
- Escrita do TCC Cap. 3 (metodologia) e Cap. 4 (resultados).

### Comandos úteis
```
.venv/bin/python inference/training/encode_text.py   # 1× (caro)
.venv/bin/python inference/training/train_rf.py
.venv/bin/python inference/training/train_gnn.py
.venv/bin/python inference/training/train_lr.py
```
