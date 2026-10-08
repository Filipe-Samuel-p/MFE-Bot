package service

import (
	"math"

	"mfebot/api/internal/model"
	"mfebot/api/internal/repository"
)

type Service struct {
	repo *repository.Repository
}

func New(repo *repository.Repository) *Service {
	return &Service{repo: repo}
}

func (s *Service) GetMetricsSummary() (*model.MetricsSummary, error) {
	rows, err := s.repo.GetRawScores()
	if err != nil {
		return nil, err
	}
	return &model.MetricsSummary{
		Total:   len(rows),
		RF:      calcMetrics(rows, func(r model.RawScoreRow) int { return binarize(r.ScoreRF) }),
		BotRGCN: calcMetrics(rows, func(r model.RawScoreRow) int { return binarize(r.ScoreGNN) }),
		MFEBot:  calcMetrics(rows, func(r model.RawScoreRow) int { return r.ClassificacaoBot }),
	}, nil
}

func (s *Service) GetEfficiency() (*model.EfficiencyStats, error) {
	rows, err := s.repo.GetEfficiencyRows()
	if err != nil {
		return nil, err
	}
	total := len(rows)
	var div, rfOnly int
	for _, r := range rows {
		switch {
		case r.FlagDivergente:
			div++
		case r.RFOnly:
			rfOnly++
		}
	}
	conv := total - div - rfOnly
	tf := float64(total)
	return &model.EfficiencyStats{
		Total:         total,
		Convergent:    conv,
		Divergent:     div,
		RFOnly:        rfOnly,
		ConvergentPct: round4(float64(conv) / tf * 100),
		DivergentPct:  round4(float64(div) / tf * 100),
		RFOnlyPct:     round4(float64(rfOnly) / tf * 100),
	}, nil
}

func (s *Service) GetDivergent() ([]model.AuditEntry, error) {
	return s.repo.GetDivergent()
}

func (s *Service) GetUser(userID string) (*model.AuditEntry, error) {
	return s.repo.GetUserByID(userID)
}

func (s *Service) GetExplanation(userID string) (*model.AuditEntryFull, error) {
	return s.repo.GetExplanationByID(userID)
}

func calcMetrics(rows []model.RawScoreRow, pred func(model.RawScoreRow) int) model.MetricSet {
	var tp, tn, fp, fn float64
	for _, r := range rows {
		p := pred(r)
		switch {
		case p == 1 && r.LabelReal == 1:
			tp++
		case p == 0 && r.LabelReal == 0:
			tn++
		case p == 1 && r.LabelReal == 0:
			fp++
		case p == 0 && r.LabelReal == 1:
			fn++
		}
	}
	prec := safeDiv(tp, tp+fp)
	rec := safeDiv(tp, tp+fn)
	return model.MetricSet{
		Accuracy:  round4((tp + tn) / (tp + tn + fp + fn)),
		Precision: round4(prec),
		Recall:    round4(rec),
		F1:        round4(safeDiv(2*prec*rec, prec+rec)),
	}
}

func binarize(score float64) int {
	if score >= 0.5 {
		return 1
	}
	return 0
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}
