"""
XAI — Explicabilidade da Random Forest via SHAP (TreeExplainer).

Módulo sob demanda: carrega o RF calibrado, extrai as features de um usuário
específico e calcula os valores SHAP para cada feature de perfil.

Por que TreeExplainer no estimador base e não no CalibratedClassifierCV:
    O SHAP TreeExplainer exige acesso à estrutura interna das árvores, o que o
    wrapper de calibração isotônica não expõe. Extraímos o RF do primeiro fold
    de validação cruzada (calibrated_classifiers_[0].estimator), que foi treinado
    com as mesmas features e na mesma ordem. A predição calibrada (score no banco)
    é reportada separadamente para contexto.

Lazy evaluation:
    1. Verifica se `explicacao_rf` já existe no banco para o user_id.
    2. Se existir → imprime e retorna sem recalcular.
    3. Se não existir → calcula SHAP, salva com UPDATE e imprime.

Uso CLI:
    python inference/xai/explain_rf.py <user_id>
"""

import json
import os
import sys
from datetime import datetime, timezone

import ijson
import joblib
import numpy as np
import shap

XAI_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(XAI_DIR)
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)
sys.path.insert(0, INFERENCE_DIR)

from db import connect  # noqa: E402
from features import FEATURE_NAMES, extract_features  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(INFERENCE_DIR, "models")
_SPLITS = ("train", "dev", "test")


def _find_user(user_id):
    """Varre os splits do TwiBot-20 em streaming e retorna o dict do usuário."""
    uid_target = str(user_id).strip()
    for split in _SPLITS:
        path = os.path.join(DATA_DIR, f"{split}.json")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            for u in ijson.items(fh, "item"):
                if str(u.get("ID", "")).strip() == uid_target:
                    return u
    return None


def _extract_sv_bot(sv_raw):
    """Extrai o vetor SHAP da classe Bot (índice 1) de qualquer formato de saída."""
    if isinstance(sv_raw, list):
        # list[class0_array, class1_array] — formato clássico do SHAP
        return np.asarray(sv_raw[1]).ravel()
    arr = np.asarray(sv_raw)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return arr[0, :, 1]
    # fallback — array 1-D ou 2-D com sample único
    return arr.ravel()


def compute_rf_explanation(user_id, rf_model=None):
    """
    Calcula a explicação SHAP para um usuário e devolve um dict JSON-serializável.

    Campos do dict:
        feature_importances  lista ordenada por |shap_value| decrescente
            .feature          nome da feature (FEATURE_NAMES)
            .shap_value       contribuição SHAP para a classe Bot
            .raw_value        valor bruto da feature (antes de qualquer normalização)
        base_value            prior do RF (E[f(x)] sobre o conjunto de treino)
        prediction_calibrated probabilidade calibrada (o score_rf gravado no banco)
        computed_at           ISO 8601 UTC
    """
    if rf_model is None:
        rf_model = joblib.load(os.path.join(MODELS_DIR, "modelo_rf.pkl"))

    user = _find_user(user_id)
    if user is None:
        raise ValueError(f"Usuário {user_id!r} não encontrado nos splits do TwiBot-20.")

    feats = np.asarray(extract_features(user), dtype=np.float64).reshape(1, -1)

    # Predição calibrada — o valor que está gravado na coluna score_rf do banco.
    p_calibrated = float(rf_model.predict_proba(feats)[0, 1])

    # RF base (pré-calibração) para o TreeExplainer.
    # CalibratedClassifierCV com cv=5 cria 5 folds; pegamos o primeiro estimador.
    base_rf = rf_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_rf)
    sv_raw = explainer.shap_values(feats)
    sv_bot = _extract_sv_bot(sv_raw)

    ev = explainer.expected_value
    base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

    feature_importances = [
        {
            "feature": name,
            "shap_value": round(float(sv_bot[i]), 6),
            "raw_value": round(float(feats[0, i]), 4),
        }
        for i, name in enumerate(FEATURE_NAMES)
    ]
    feature_importances.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "feature_importances": feature_importances,
        "base_value": round(base_value, 6),
        "prediction_calibrated": round(p_calibrated, 6),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def save_to_db(user_id, explanation, conn=None):
    """Grava explicacao_rf no banco via UPDATE (idempotente)."""
    close_after = conn is None
    if close_after:
        conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auditoria_deteccao_bots SET explicacao_rf = %s WHERE user_id = %s",
                (json.dumps(explanation, ensure_ascii=False), str(user_id)),
            )
        conn.commit()
    finally:
        if close_after:
            conn.close()


def explain_lazy(user_id, conn=None):
    """
    Lazy evaluation: calcula e persiste apenas se ainda não existe no banco.
    Retorna o dict de explicação (seja do cache ou recém-calculado).
    """
    close_after = conn is None
    if close_after:
        conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT explicacao_rf FROM auditoria_deteccao_bots WHERE user_id = %s",
                (str(user_id),),
            )
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"user_id={user_id!r} não encontrado no banco.")
        if row[0] is not None:
            return row[0]  # psycopg2 deserializa JSONB automaticamente
        exp = compute_rf_explanation(user_id)
        save_to_db(user_id, exp, conn=conn)
        return exp
    finally:
        if close_after:
            conn.close()


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python explain_rf.py <user_id>")
    user_id = sys.argv[1]

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT explicacao_rf FROM auditoria_deteccao_bots WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()

        if row is None:
            sys.exit(f"user_id={user_id!r} não encontrado na tabela auditoria_deteccao_bots.")

        if row[0] is not None:
            print(f"[cache] Explicação RF já existe para user_id={user_id!r}:")
            print(json.dumps(row[0], indent=2, ensure_ascii=False))
            return

        print(f"Calculando SHAP (TreeExplainer) para user_id={user_id!r}...")
        exp = compute_rf_explanation(user_id)
        save_to_db(user_id, exp, conn=conn)
        print("Salvo em explicacao_rf:")
        print(json.dumps(exp, indent=2, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
