"""
Parâmetros globais do projeto. Altere aqui para mudar comportamento
sem precisar editar os módulos individuais.
"""

from datetime import datetime

## Período dos dados
START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

## Ações por setor
TICKERS_BY_SECTOR: dict[str, list[str]] = {
    "MINERACAO E SIDERURGIA": ["VALE3.SA", "CMIN3.SA", "GGBR4.SA", "USIM5.SA", "CSNA3.SA"],
    "FINANCEIRAS": ["ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "BPAC11.SA", "B3SA3.SA"],
    "PETROLEO, GAS E BIOCOMBUSTIVEIS": ["PETR4.SA", "PETR3.SA", "PRIO3.SA", "UGPA3.SA"],
    "ALIMENTOS E BEBIDAS": ["BRFS3.SA", "CRFB3.SA", "JBSS3.SA"],
    "UTILIDADE PUBLICA": ["AXIA3.SA", "SBSP3.SA"],
    "INDUSTRIA E CONSUMO": ["LREN3.SA", "RADL3.SA", "MGLU3.SA", "RENT3.SA", "EMBR3.SA", "CBAV3.SA"],
}

## Mega caps a serem removidas
MEGA_CAPS: list[str] = ["PETR4", "PETR3", "VALE3", "ITUB4", "BBDC4", "BBAS3"]

## Ticker do índice de referência
IBOV_TICKER = "^BVSP"

## Definição do target
TARGET_HORIZON = 5 # dias úteis à frente
TARGET_THRESHOLD = 0.05 # retorno mínimo para classe positiva (5%)

## Split de treino e teste
TRAIN_RATIO = 0.8 # 80% das datas para treino

## Parâmetros dos modelos
RANDOM_STATE = 42

RF_PARAMS = dict(
    n_estimators = 200,
    max_depth = 5,
    min_samples_leaf = 200,
    class_weight = 'balanced',
    random_state = RANDOM_STATE,
    n_jobs = -1
)

XGB_BASE_PARAMS = dict(
    n_estimators    = 200,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.7,
    eval_metric     = "logloss",
    random_state    = RANDOM_STATE,
    n_jobs          = -1
)

XGB_TUNED_N_ESTIMATORS = 1000
XGB_EARLY_STOPPING_ROUNDS = 30
XGB_FINAL_VAL_RATIO = 0.1

XGB_PARAM_DIST = {
    "max_depth":        [3, 4, 5, 6],
    "min_child_weight": [1, 5, 10, 20],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "n_estimators":     [200, 300, 500],
    "subsample":        [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "reg_alpha":        [0, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}

RANDOM_SEARCH_N_ITER = 40

## Paths de saida
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/'