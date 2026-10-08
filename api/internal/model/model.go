package model

import (
	"encoding/json"
	"fmt"
	"time"
)

// NullRawJSON faz scan de colunas JSONB anuláveis do PostgreSQL (pq driver retorna nil para NULL).
type NullRawJSON struct {
	Data  json.RawMessage
	Valid bool
}

func (n *NullRawJSON) Scan(value any) error {
	if value == nil {
		n.Data, n.Valid = nil, false
		return nil
	}
	n.Valid = true
	switch v := value.(type) {
	case []byte:
		n.Data = make(json.RawMessage, len(v))
		copy(n.Data, v)
	case string:
		n.Data = json.RawMessage(v)
	default:
		return fmt.Errorf("NullRawJSON: tipo inesperado %T", value)
	}
	return nil
}

func (n NullRawJSON) MarshalJSON() ([]byte, error) {
	if !n.Valid || n.Data == nil {
		return []byte("null"), nil
	}
	return n.Data, nil
}

// MetricSet reúne as 4 métricas de classificação de um modelo.
type MetricSet struct {
	Accuracy  float64 `json:"accuracy"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

// MetricsSummary é a resposta de GET /v1/metrics/summary.
type MetricsSummary struct {
	Total   int       `json:"total"`
	RF      MetricSet `json:"rf"`
	BotRGCN MetricSet `json:"botrgcn"`
	MFEBot  MetricSet `json:"mfe_bot"`
}

// EfficiencyStats é a resposta de GET /v1/metrics/efficiency.
type EfficiencyStats struct {
	Total         int     `json:"total"`
	Convergent    int     `json:"convergent"`
	Divergent     int     `json:"divergent"`
	RFOnly        int     `json:"rf_only"`
	ConvergentPct float64 `json:"convergent_pct"`
	DivergentPct  float64 `json:"divergent_pct"`
	RFOnlyPct     float64 `json:"rf_only_pct"`
}

// AuditEntry é a linha principal da tabela auditoria_deteccao_bots (sem JSONB).
// Usada em GET /v1/audit/divergent e GET /v1/audit/users/{id}.
type AuditEntry struct {
	ID               int       `db:"id"                json:"id"`
	UserID           string    `db:"user_id"           json:"user_id"`
	ScreenName       *string   `db:"screen_name"       json:"screen_name"`
	ScoreRF          float64   `db:"score_rf"          json:"score_rf"`
	ScoreGNN         float64   `db:"score_gnn"         json:"score_gnn"`
	ScoreMeta        *float64  `db:"score_meta"        json:"score_meta"`
	FlagDivergente   bool      `db:"flag_divergente"   json:"flag_divergente"`
	RFOnly           bool      `db:"rf_only"           json:"rf_only"`
	VeredutoFinal    float64   `db:"veredito_final"    json:"veredito_final"`
	ClassificacaoBot int       `db:"classificacao_bot" json:"classificacao_bot"`
	LabelReal        *int      `db:"label_real"        json:"label_real"`
	DataAnalise      time.Time `db:"data_analise"      json:"data_analise"`
}

// AuditEntryFull inclui os campos JSONB de explicabilidade.
// Usada em GET /v1/audit/explanation/{id}.
type AuditEntryFull struct {
	ID               int             `db:"id"                json:"id"`
	UserID           string          `db:"user_id"           json:"user_id"`
	ScreenName       *string         `db:"screen_name"       json:"screen_name"`
	ScoreRF          float64         `db:"score_rf"          json:"score_rf"`
	ScoreGNN         float64         `db:"score_gnn"         json:"score_gnn"`
	ScoreMeta        *float64        `db:"score_meta"        json:"score_meta"`
	FlagDivergente   bool            `db:"flag_divergente"   json:"flag_divergente"`
	RFOnly           bool            `db:"rf_only"           json:"rf_only"`
	VeredutoFinal    float64         `db:"veredito_final"    json:"veredito_final"`
	ClassificacaoBot int             `db:"classificacao_bot" json:"classificacao_bot"`
	LabelReal        *int            `db:"label_real"        json:"label_real"`
	ExplicacaoRF     NullRawJSON `db:"explicacao_rf"     json:"explicacao_rf"`
	ExplicacaoGNN    NullRawJSON `db:"explicacao_gnn"    json:"explicacao_gnn"`
	DataAnalise      time.Time       `db:"data_analise"      json:"data_analise"`
}

// Tipos internos usados apenas para cálculo de métricas e eficiência.

type RawScoreRow struct {
	ScoreRF          float64 `db:"score_rf"`
	ScoreGNN         float64 `db:"score_gnn"`
	ClassificacaoBot int     `db:"classificacao_bot"`
	LabelReal        int     `db:"label_real"`
}

type EfficiencyRow struct {
	FlagDivergente bool `db:"flag_divergente"`
	RFOnly         bool `db:"rf_only"`
}
