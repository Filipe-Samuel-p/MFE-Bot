"""
Roteamento por Cruzamento de Fronteira (Camada 2 — mecanismo central do MFE-Bot).

Lógica puramente semântica (TCC §3.4.2, Tabela 3): cada especialista converte
seu score em classe pela fronteira 0.5; comparam-se as duas classes.
- CONVERGÊNCIA (mesma classe) → Early Exit: média aritmética dos scores.
- DIVERGÊNCIA (classes opostas) → aciona a Regressão Logística (arbitragem).

Caso especial (regra 4): usuário `neighbor:null` não entra no stacking — só o RF
decide (RF-only). O score do GNN ainda é gravado para auditoria, mas não roteia.

`veredito_final` é sempre probabilidade bruta; `classificacao_bot` é derivada.
"""

from collections import namedtuple

RouteResult = namedtuple(
    "RouteResult",
    ["flag_divergente", "rf_only", "score_meta", "veredito_final", "classificacao_bot"],
)


def route(p_rf, p_gnn, lr_model, has_neighbor=True):
    """Aplica o cruzamento de fronteira e devolve o desfecho do roteador."""
    if not has_neighbor:
        # RF-only: fora do stacking. GNN não participa da decisão.
        # Categoria persistida explicitamente (rf_only), não inferida a
        # posteriori por comparação de floats.
        veredito = float(p_rf)
        return RouteResult(False, True, None, veredito, _bin(veredito))

    pred_rf = p_rf >= 0.5
    pred_gnn = p_gnn >= 0.5

    if pred_rf == pred_gnn:
        # Convergência — Early Exit pela média.
        veredito = (float(p_rf) + float(p_gnn)) / 2.0
        return RouteResult(False, False, None, veredito, _bin(veredito))

    # Divergência — arbitragem pela Regressão Logística.
    score_meta = float(lr_model.predict_proba([[p_rf, p_gnn]])[0][1])
    return RouteResult(True, False, score_meta, score_meta, _bin(score_meta))


def _bin(veredito):
    return 1 if veredito >= 0.5 else 0
