package repository

import (
	"github.com/jmoiron/sqlx"

	"mfebot/api/internal/model"
)

type Repository struct {
	db *sqlx.DB
}

func New(db *sqlx.DB) *Repository {
	return &Repository{db: db}
}

func (r *Repository) GetRawScores() ([]model.RawScoreRow, error) {
	var rows []model.RawScoreRow
	err := r.db.Select(&rows,
		`SELECT score_rf, score_gnn, classificacao_bot, label_real
		 FROM auditoria_deteccao_bots
		 WHERE label_real IS NOT NULL`)
	return rows, err
}

func (r *Repository) GetEfficiencyRows() ([]model.EfficiencyRow, error) {
	var rows []model.EfficiencyRow
	err := r.db.Select(&rows,
		`SELECT flag_divergente, rf_only
		 FROM auditoria_deteccao_bots`)
	return rows, err
}

func (r *Repository) GetDivergent() ([]model.AuditEntry, error) {
	var rows []model.AuditEntry
	err := r.db.Select(&rows,
		`SELECT id, user_id, screen_name, score_rf, score_gnn, score_meta,
		        flag_divergente, rf_only, veredito_final, classificacao_bot, label_real, data_analise
		 FROM auditoria_deteccao_bots
		 WHERE flag_divergente = TRUE
		 ORDER BY id`)
	return rows, err
}

func (r *Repository) GetUserByID(userID string) (*model.AuditEntry, error) {
	var row model.AuditEntry
	err := r.db.Get(&row,
		`SELECT id, user_id, screen_name, score_rf, score_gnn, score_meta,
		        flag_divergente, rf_only, veredito_final, classificacao_bot, label_real, data_analise
		 FROM auditoria_deteccao_bots
		 WHERE user_id = $1`,
		userID)
	if err != nil {
		return nil, err
	}
	return &row, nil
}

func (r *Repository) GetExplanationByID(userID string) (*model.AuditEntryFull, error) {
	var row model.AuditEntryFull
	err := r.db.Get(&row,
		`SELECT id, user_id, screen_name, score_rf, score_gnn, score_meta,
		        flag_divergente, rf_only, veredito_final, classificacao_bot, label_real,
		        explicacao_rf, explicacao_gnn, data_analise
		 FROM auditoria_deteccao_bots
		 WHERE user_id = $1`,
		userID)
	if err != nil {
		return nil, err
	}
	return &row, nil
}
