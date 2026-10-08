package handler

import (
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"

	"mfebot/api/internal/service"
)

type Handler struct {
	svc *service.Service
}

func New(svc *service.Service) *Handler {
	return &Handler{svc: svc}
}

// GET /v1/metrics/summary
// Retorna Acurácia, Precisão, Revocação e F1 para RF, BotRGCN e MFE-Bot.
func (h *Handler) MetricsSummary(w http.ResponseWriter, r *http.Request) {
	data, err := h.svc.GetMetricsSummary()
	respond(w, data, err)
}

// GET /v1/metrics/efficiency
// Retorna contagem de convergentes, divergentes e RF-only no test.json.
func (h *Handler) MetricsEfficiency(w http.ResponseWriter, r *http.Request) {
	data, err := h.svc.GetEfficiency()
	respond(w, data, err)
}

// GET /v1/audit/divergent
// Lista todos os usuários com flag_divergente = TRUE.
func (h *Handler) AuditDivergent(w http.ResponseWriter, r *http.Request) {
	data, err := h.svc.GetDivergent()
	respond(w, data, err)
}

// GET /v1/audit/users/{id}
// Diagnóstico completo de um usuário (sem JSONB de XAI).
func (h *Handler) AuditUser(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	data, err := h.svc.GetUser(id)
	if errors.Is(err, sql.ErrNoRows) {
		writeError(w, http.StatusNotFound, "user not found")
		return
	}
	respond(w, data, err)
}

// GET /v1/audit/explanation/{id}
// Retorna o diagnóstico completo + campos JSONB de XAI (null até o módulo XAI rodar).
func (h *Handler) AuditExplanation(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	data, err := h.svc.GetExplanation(id)
	if errors.Is(err, sql.ErrNoRows) {
		writeError(w, http.StatusNotFound, "user not found")
		return
	}
	respond(w, data, err)
}

func respond(w http.ResponseWriter, data any, err error) {
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
