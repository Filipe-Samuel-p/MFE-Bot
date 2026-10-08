"""
Extração de features tabulares de perfil para a Random Forest.

Módulo COMPARTILHADO entre o treino (training/train_rf.py) e o worker de
inferência (worker/run_inference.py). Manter a extração em um único lugar
evita divergência train/serve: o modelo precisa ver exatamente as mesmas
colunas, na mesma ordem, com a mesma normalização.

Particularidades do TwiBot-20 tratadas aqui:
- Todo valor de `profile` vem como string com espaço no final ('15349596 ').
- Booleanos vêm como texto: 'True ', 'False ', 'None '.
- Campos ausentes ou 'None' são tratados como neutro (0).
- `profile` pode estar ausente; `neighbor` pode ser None (não afeta o RF).
"""

from datetime import datetime, timezone

# Data de referência para calcular a idade da conta.
# O TwiBot-20 foi coletado em meados de 2020; usamos uma data fixa para que
# a feature seja reprodutível (datetime.now() mudaria a cada execução).
REFERENCE_DATE = datetime(2020, 9, 28, tzinfo=timezone.utc)

_CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"

# Atributos numéricos (contagens, razões, datas, comprimentos). Para o GNN
# entram como "num_prop" (z-score); para o RF, parte do vetor tabular único.
NUMERIC_FEATURE_NAMES = [
    "followers_count",
    "friends_count",
    "listed_count",
    "favourites_count",
    "statuses_count",
    "followers_friends_ratio",
    "account_age_days",
    "statuses_per_day",
    "screen_name_length",
    "name_length",
    "description_length",
    "num_digits_screen_name",
]

# Atributos categóricos/booleanos (0/1). Para o GNN entram como "cat_prop".
CATEGORICAL_FEATURE_NAMES = [
    "protected",
    "verified",
    "geo_enabled",
    "default_profile",
    "default_profile_image",
    "profile_use_background_image",
]

# Ordem canônica das colunas (numéricas primeiro, depois categóricas).
# NÃO reordenar sem retreinar o modelo. O RF usa este vetor completo; o GNN
# usa o particionamento numeric/categorical acima.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
N_NUMERIC = len(NUMERIC_FEATURE_NAMES)


def _clean(value):
    """Normaliza um valor cru do TwiBot-20: strip + 'None'/'' viram None."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none" or s.lower() == "null":
        return None
    return s


def _to_float(value):
    s = _clean(value)
    if s is None:
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _to_bool(value):
    """'True ' -> 1.0, qualquer outra coisa -> 0.0."""
    s = _clean(value)
    return 1.0 if (s is not None and s.lower() == "true") else 0.0


def _account_age_days(created_at):
    s = _clean(created_at)
    if s is None:
        return 0.0
    try:
        created = datetime.strptime(s, _CREATED_AT_FMT)
    except (ValueError, TypeError):
        return 0.0
    delta = (REFERENCE_DATE - created).days
    return float(max(delta, 0))


def _str_len(value):
    s = _clean(value)
    return float(len(s)) if s is not None else 0.0


def _num_digits(value):
    s = _clean(value)
    if s is None:
        return 0.0
    return float(sum(c.isdigit() for c in s))


def extract_features(user):
    """
    Recebe um registro de usuário do TwiBot-20 e devolve uma lista de floats
    na ordem de FEATURE_NAMES.

    Aceita usuários com `profile` ausente (devolve vetor neutro de zeros) e
    ignora `neighbor`/`tweet` — o RF usa apenas o perfil tabular.
    """
    profile = user.get("profile") or {}

    followers = _to_float(profile.get("followers_count"))
    friends = _to_float(profile.get("friends_count"))
    listed = _to_float(profile.get("listed_count"))
    favourites = _to_float(profile.get("favourites_count"))
    statuses = _to_float(profile.get("statuses_count"))

    age_days = _account_age_days(profile.get("created_at"))

    followers_friends_ratio = followers / (friends + 1.0)
    statuses_per_day = statuses / (age_days + 1.0)

    return [
        followers,
        friends,
        listed,
        favourites,
        statuses,
        followers_friends_ratio,
        age_days,
        statuses_per_day,
        _str_len(profile.get("screen_name")),
        _str_len(profile.get("name")),
        _str_len(profile.get("description")),
        _num_digits(profile.get("screen_name")),
        _to_bool(profile.get("protected")),
        _to_bool(profile.get("verified")),
        _to_bool(profile.get("geo_enabled")),
        _to_bool(profile.get("default_profile")),
        _to_bool(profile.get("default_profile_image")),
        _to_bool(profile.get("profile_use_background_image")),
    ]


def parse_label(user):
    """Converte o label do TwiBot-20 ('0'/'1') em int. Retorna None se ausente."""
    raw = _clean(user.get("label"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None
