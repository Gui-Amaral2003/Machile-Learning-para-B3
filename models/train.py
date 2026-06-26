"""
Treinamento dos modelos: Random Forest (baseline), XGBoost sem tuning
e XGBoost tunado com PurgedTimeSeriesSplit + RandomizedSearchCV.
 
Todas as funções recebem dados já normalizados (X_train_s, X_test_s)
e retornam os modelos treinados prontos para avaliação.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import sys
import os
sys.path.append(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RANDOM_STATE, RF_PARAMS, XGB_BASE_PARAMS,
    XGB_TUNED_N_ESTIMATORS, XGB_EARLY_STOPPING_ROUNDS,
    XGB_FINAL_VAL_RATIO, XGB_PARAM_DIST, RANDOM_SEARCH_N_ITER 
)
from cv import PurgedTimeSeriesSplit

logger = logging.getLogger(__name__)

def _scale_pos_weight(y_train: pd.Series) -> float:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    logger.info(f"scale_pos_weight = {spw}  ({neg}, {pos})")
    return spw

def train_random_forest(X_train: np.ndarray,y_train: pd.Series) -> RandomForestClassifier:
    logger.info("Treinando Random Forest")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    logger.info("Random Forest treinado")
    return model

def train_xgboost_base(X_train: np.ndarray,y_train: pd.Series) -> XGBClassifier:
    """
    XGBoost com parâmetros fixos definidos em config.py.
    Não usa X_test no fit — avaliação é feita separadamente em evaluate.py.
    """
    logger.info("Treinando XGBoost base")
    spw = _scale_pos_weight(y_train)
    xgb = XGBClassifier(**XGB_BASE_PARAMS, scale_pos_weight=spw)
    xgb.fit(X_train, y_train, verbose=False)
    logger.info("XGBoost base treinado")
    return xgb

def tune_xgboost(X_train: np.ndarray,y_train: pd.Series,train_dates: pd.DatetimeIndex) -> tuple[XGBClassifier, dict]:
    """
    Busca hiperparâmetros via RandomizedSearchCV com PurgedTimeSeriesSplit e re-treina o modelo final com early stopping.
 
    Returns
    (modelo_final, melhores_params)
    """
    logger.info(f"Iniciando tuning do XGBoost (n_iter={RANDOM_SEARCH_N_ITER})")
    spw = _scale_pos_weight(y_train)
 
    xgb_base = XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
 
    cv     = PurgedTimeSeriesSplit()
    search = RandomizedSearchCV(
        estimator = xgb_base,
        param_distributions = XGB_PARAM_DIST,
        n_iter = RANDOM_SEARCH_N_ITER,
        cv = cv,
        scoring = "f1",
        n_jobs = -1,
        verbose = 1,
        random_state = RANDOM_STATE
    )
    search.fit(X_train, y_train, groups=train_dates)
 
    logger.info(f"Melhor F1 na CV: {search.best_score_}" )
    logger.info(f"Melhores parâmetros: {search.best_params_}")
 
    ## Confirma que não há vazamento entre folds
    _log_cv_diagnostic(cv, X_train, train_dates)
 
    ## Re-treina com n_estimators maior + early stopping
    best_params = search.best_params_.copy()
    best_params["n_estimators"] = XGB_TUNED_N_ESTIMATORS
 
    val_size = int(len(X_train) * XGB_FINAL_VAL_RATIO)
    X_tr2, y_tr2 = X_train[:-val_size], y_train.iloc[:-val_size]
    X_val2, y_val2 = X_train[-val_size:], y_train.iloc[-val_size:]
 
    xgb_tuned = XGBClassifier(
        **best_params,
        scale_pos_weight = spw,
        eval_metric = "logloss",
        random_state = RANDOM_STATE,
        n_jobs = -1,
        early_stopping_rounds = XGB_EARLY_STOPPING_ROUNDS,
    )
    xgb_tuned.fit(X_tr2, y_tr2, eval_set=[(X_val2, y_val2)], verbose=False)
    logger.info(f"XGBoost tunado: {xgb_tuned.best_iteration} árvores usadas de {XGB_TUNED_N_ESTIMATORS}",)
 
    return xgb_tuned, search.best_params_

def _log_cv_diagnostic(cv: PurgedTimeSeriesSplit,X_train: np.ndarray,train_dates: pd.DatetimeIndex) -> None:
    logger.info("Diagnóstico do Purged CV")
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, groups=train_dates)):
        tr_dates  = train_dates[tr_idx]
        val_dates = train_dates[val_idx]
        overlap   = val_dates.min() < tr_dates.max()
        status    = "VAZAMENTO" if overlap else "OK"
        logger.info(
            f"Fold {fold + 1}: treino até {tr_dates.max.date()} | val {val_dates.min().date()}->{val_dates.max().date()} | {status}")