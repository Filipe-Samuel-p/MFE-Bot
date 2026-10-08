CREATE TABLE auditoria_deteccao_bots (
    id                SERIAL PRIMARY KEY,
    user_id           VARCHAR(50)  NOT NULL,
    screen_name       VARCHAR(100),
    score_rf          FLOAT        NOT NULL,
    score_gnn         FLOAT        NOT NULL,
    score_meta        FLOAT,
    flag_divergente   BOOLEAN      NOT NULL,
    rf_only           BOOLEAN      NOT NULL DEFAULT FALSE,
    veredito_final    FLOAT        NOT NULL,
    classificacao_bot INTEGER      NOT NULL,
    label_real        INTEGER,
    explicacao_rf     JSONB,
    explicacao_gnn    JSONB,
    data_analise      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_id           ON auditoria_deteccao_bots(user_id);
CREATE INDEX idx_flag_divergente   ON auditoria_deteccao_bots(flag_divergente);
CREATE INDEX idx_label_real        ON auditoria_deteccao_bots(label_real);
CREATE INDEX idx_classificacao_bot ON auditoria_deteccao_bots(classificacao_bot);
