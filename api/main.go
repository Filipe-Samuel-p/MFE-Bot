package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	_ "github.com/lib/pq"
	"github.com/jmoiron/sqlx"

	"mfebot/api/internal/handler"
	"mfebot/api/internal/repository"
	"mfebot/api/internal/service"
)

func main() {
	db, err := sqlx.Connect("postgres", dsn())
	if err != nil {
		log.Fatalf("db connection: %v", err)
	}
	defer db.Close()

	repo := repository.New(db)
	svc := service.New(repo)
	h := handler.New(svc)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/metrics/summary", h.MetricsSummary)
	mux.HandleFunc("GET /v1/metrics/efficiency", h.MetricsEfficiency)
	mux.HandleFunc("GET /v1/audit/divergent", h.AuditDivergent)
	mux.HandleFunc("GET /v1/audit/users/{id}", h.AuditUser)
	mux.HandleFunc("GET /v1/audit/explanation/{id}", h.AuditExplanation)

	addr := ":" + getenv("PORT", "8080")
	log.Printf("MFE-Bot API em %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func dsn() string {
	return fmt.Sprintf(
		"host=%s port=%s dbname=%s user=%s password=%s sslmode=disable",
		getenv("DB_HOST", "localhost"),
		getenv("DB_PORT", "5432"),
		getenv("DB_NAME", "mfebot"),
		getenv("DB_USER", "mfebot"),
		getenv("DB_PASSWORD", "mfebot"),
	)
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
