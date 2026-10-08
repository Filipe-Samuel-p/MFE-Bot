"""
Testes do Roteamento por Cruzamento de Fronteira (TCC Cap. 3, Tabela 3).

Cobre os quatro desfechos do roteador (convergência bot/humano, divergência
nos dois sentidos) mais o caso especial RF-only (neighbor:null). Sem
dependências externas: usa unittest (stdlib) e um stub de LogisticRegression.

Uso:
    .venv/bin/python -m unittest inference.tests.test_router -v
"""

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(INFERENCE_DIR, "worker"))

from router import route  # noqa: E402


class StubLR:
    """Substitui a Regressão Logística real: devolve um score fixo, para que
    o teste do roteador não dependa de nenhum modelo treinado."""

    def __init__(self, p_bot):
        self.p_bot = p_bot

    def predict_proba(self, X):
        return [[1.0 - self.p_bot, self.p_bot]]


class TestRoteadorCruzamentoDeFronteira(unittest.TestCase):
    def test_convergencia_ambos_bot(self):
        r = route(p_rf=0.9, p_gnn=0.8, lr_model=None)
        self.assertFalse(r.flag_divergente)
        self.assertFalse(r.rf_only)
        self.assertIsNone(r.score_meta)
        self.assertAlmostEqual(r.veredito_final, 0.85)
        self.assertEqual(r.classificacao_bot, 1)

    def test_convergencia_ambos_humano(self):
        r = route(p_rf=0.2, p_gnn=0.1, lr_model=None)
        self.assertFalse(r.flag_divergente)
        self.assertFalse(r.rf_only)
        self.assertIsNone(r.score_meta)
        self.assertAlmostEqual(r.veredito_final, 0.15)
        self.assertEqual(r.classificacao_bot, 0)

    def test_divergencia_rf_bot_gnn_humano_lr_decide_bot(self):
        lr = StubLR(p_bot=0.7)
        r = route(p_rf=0.8, p_gnn=0.3, lr_model=lr)
        self.assertTrue(r.flag_divergente)
        self.assertFalse(r.rf_only)
        self.assertAlmostEqual(r.score_meta, 0.7)
        self.assertAlmostEqual(r.veredito_final, 0.7)
        self.assertEqual(r.classificacao_bot, 1)

    def test_divergencia_rf_humano_gnn_bot_lr_decide_humano(self):
        lr = StubLR(p_bot=0.4)
        r = route(p_rf=0.3, p_gnn=0.9, lr_model=lr)
        self.assertTrue(r.flag_divergente)
        self.assertFalse(r.rf_only)
        self.assertAlmostEqual(r.score_meta, 0.4)
        self.assertAlmostEqual(r.veredito_final, 0.4)
        self.assertEqual(r.classificacao_bot, 0)

    def test_fronteira_exata_0_5_conta_como_bot(self):
        # pred = p >= 0.5 (não p > 0.5); os dois convergem em "bot".
        r = route(p_rf=0.5, p_gnn=0.5, lr_model=None)
        self.assertFalse(r.flag_divergente)
        self.assertFalse(r.rf_only)
        self.assertAlmostEqual(r.veredito_final, 0.5)
        self.assertEqual(r.classificacao_bot, 1)

    def test_rf_only_neighbor_null_ignora_gnn_e_lr(self):
        # lr_model=None de propósito: RF-only nunca deve tocar no meta-classificador.
        r = route(p_rf=0.7, p_gnn=0.1, lr_model=None, has_neighbor=False)
        self.assertFalse(r.flag_divergente)
        self.assertTrue(r.rf_only)
        self.assertIsNone(r.score_meta)
        self.assertAlmostEqual(r.veredito_final, 0.7)
        self.assertEqual(r.classificacao_bot, 1)

    def test_rf_only_humano(self):
        r = route(p_rf=0.2, p_gnn=0.9, lr_model=None, has_neighbor=False)
        self.assertFalse(r.flag_divergente)
        self.assertTrue(r.rf_only)
        self.assertIsNone(r.score_meta)
        self.assertAlmostEqual(r.veredito_final, 0.2)
        self.assertEqual(r.classificacao_bot, 0)


if __name__ == "__main__":
    unittest.main()
