"""
Responsavel exclusivamente por coletar os dados de cada ativo via yfinance.
Retorna dataframes crus, sem nenhum tipo de feature engineering
"""

import logging
import pandas as pd
import yfinance as yf
from curl_cffi import requests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    START_DATE, END_DATE,
    TICKERS_BY_SECTOR, MEGA_CAPS,
    IBOV_TICKER,
)

logger = logging.getLogger(__name__)

# Sessão com impersonação de Chrome para evitar rate-limiting do Yahoo Finance
_session = requests.Session(impersonate="chrome")

def _download_single_ticker(ticker: str, sector: str) -> pd.DataFrame | None:
    """
    Baixa dados de um único ticker e adiciona colunas de metadado.
    Retorna None se o download resultar em DataFrame vazio.
    """
    df = yf.download(
        ticker,
        start = START_DATE,
        end = END_DATE,
        auto_adjust = True,
        progress = False,
        session = _session
    )

    if df.empty:
        logger.warning(f"Ticker {ticker} retornou DataFrame vazio. Ignorando")
        return None
    
    ## Resolvendo MultiIndex nas colunas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df["ticker"] = ticker.replace(".SA", "")
    df["sector"] = sector

    return df

def download_ibov() -> pd.DataFrame:
    """
    Baixa o indice Ibovespa e retorna um DF com coluna 'ibov_ret', indexado por Date
    """
    logger.info(f"Baixando Ibovespa {IBOV_TICKER}")
    ibov = yf.download(
        IBOV_TICKER,
        start = START_DATE,
        end = END_DATE,
        auto_adjust = True,
        progress = False,
        session = _session
    )

    if isinstance(ibov.columns, pd.MultiIndex):
        ibov.columns = ibov.columns.get_level_values(0)
    
    ibov_ret = ibov['Close'].pct_change().rename('ibov_ret')
    return ibov_ret.to_frame()


