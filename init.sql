CREATE TABLE deteccao_bots (
    id               SERIAL PRIMARY KEY,
    user_id          VARCHAR(50)   NOT NULL,
    screen_name      VARCHAR(100),
    p_rf             FLOAT         NOT NULL,
    p_gnn            FLOAT         NOT NULL,
    p_meta           FLOAT,
    flag_divergente  BOOLEAN       NOT NULL,
    veredito_final   FLOAT         NOT NULL,
    label_real       INTEGER       NOT NULL,
    data_analise     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_id         ON deteccao_bots(user_id);
CREATE INDEX idx_flag_divergente ON deteccao_bots(flag_divergente);
CREATE INDEX idx_label_real      ON deteccao_bots(label_real);
