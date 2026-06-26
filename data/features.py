"""
Toda a engenharia de features (indicadores técnicos, alpha/beta vs IBOV,
sazonalidade) e definição do target binário.
 
Convenção importante: nenhuma feature olha para o futuro.
O shift(-N) só é usado para calcular o *target*, e o resultado é
atribuído à linha do dia t (não do dia t+N).
"""

import logging
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TARGET_HORIZON, TARGET_THRESHOLD

logger = logging.getLogger(__name__)

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta).where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame, ibov_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o DataFrame bruto de ações e o DataFrame do Ibov e devolve o DF enriquecido com todas as features e o target.

    Params:
    df: DataFrame empilhado (Date, Open, High, Low, Close, Volume, ticker, sector)
    ibov_ret: DataFrame com coluna 'ibov_ret' indexado por Date

    Returns:
    DataFrame com features calculadas, target e limpo
    """
    df = df.copy()

    ## Retornos passados
    logger.info("Calculando retornos passados")
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"Return_{n}"] = df.groupby("ticker")['Close'].transform(lambda x: x.pct_change(n))

    ## Médias móveis (razão x preço)
    logger.info("Calculando médias móveis")
    for w in [5, 10, 20, 50]:
        df[f"sma_ratio_{w}"] = df.groupby("ticker")["Close"].transform(lambda x: x / x.rolling(w).mean() - 1)

    ## RSI
    logger.info("Calculando RSI") 
    df["rsi"] = df.groupby("ticker")["Close"].transform(lambda x: _rsi(x, period=14))

    ## MACD
    logger.info("Calculando MACD")
    df["macd"] = df.groupby("ticker")["Close"].transform(lambda x: (x.ewm(span=12).mean() - x.ewm(span=26).mean()) / x)
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    ## Volatilidade e volume
    logger.info("Calculando volatilidade e volume")
    df['vol_20'] = df.groupby("ticker")['return_1'].transform(lambda x: x.rolling(20).std())
    df['vol_ratio'] = df.groupby("ticker")['vol_20'].transform(lambda x: x / x.rolling(20).mean() - 1)
    df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["volatility_change"] = df.groupby("ticker")["vol_20"].transform(lambda x: x / x.rolling(20).mean() - 1)
    df["volume_spike"] = df.groupby("ticker")["Volume"].transform(lambda x: (x > x.rolling(20).mean() * 2).astype(int))

    ## Tendências curto vs longo prazo
    logger.info("Calculando tendências SMA50/SMA200")
    sma50 = df.groupby("ticker")["Close"].transform(lambda x: x.rolling(50).mean())
    sma200 = df.groupby("ticker")["Close"].transform(lambda x: x.rolling(200).mean())
    df["trend_50_200"] = (sma50 > sma200).astype(int)
    df["above_sma_200"] = (df["Close"] > sma200).astype(int)
    df["momentum_20"] = df.groupby("ticker")["Close"].transform(lambda x: x / x.shift(20) - 1)

    ## Features relativas ao IBOV
    logger.info("Calculando alpha e beta vs. IBOV")
    df = df.join(ibov_ret["ibov_ret"], on="Date")
 
    df["alpha_1d"] = df["return_1"] - df["ibov_ret"]
    df["alpha_5d"] = df["return_5"]  - df.groupby("ticker")["ibov_ret"].transform(lambda x: x.rolling(5).sum())
    df["alpha_20d"] = df["return_20"] - df.groupby("ticker")["ibov_ret"].transform(lambda x: x.rolling(20).sum())
 
    ibov_vol = ibov_ret["ibov_ret"].rolling(20).std()
    df["ibov_vol"] = df["Date"].map(ibov_vol)
 
    df["beta_20"] = (
        df.groupby("ticker")
        .apply(
            lambda g: g["return_1"].rolling(20).cov(g["ibov_ret"])
            / g["ibov_ret"].rolling(20).var()
        )
        .reset_index(level=0, drop=True)
    )

    ## Extras: z-score, distÂncia de 52 semanas, dia da semana
    logger.info("Calculando z-score, extremos de 52 semanas e DOW ...")
    roll_mean_20 = df.groupby("ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    roll_std_20  = df.groupby("ticker")["Close"].transform(lambda x: x.rolling(20).std())
    df["zscore_20"]  = (df["Close"] - roll_mean_20) / roll_std_20
 
    df["dist_max52"] = df.groupby("ticker")["Close"].transform(lambda x: x / x.rolling(252).max() - 1)
    df["dist_min52"] = df.groupby("ticker")["Close"].transform(lambda x: x / x.rolling(252).min() - 1)
    df["dow"] = pd.to_datetime(df["Date"]).dt.dayofweek

    ## Setor como dummy
    df = pd.get_dummies(df, columns = ['sector'], prefix = 'sec', dtype = int)

    ## Target
    logger.info(f"Calculando target: retorno >= {TARGET_THRESHOLD * 100} em {TARGET_HORIZON} dias")

    ret_fut = df.groupby("ticker")['Close'].transform(lambda x: x.shift(-TARGET_HORIZON) / x - 1)

    df['target'] = (ret_fut >= TARGET_THRESHOLD).astype(int)

    before = len(df)
    df = df.dropna()
    logger.info(f"Linhas removidas por NaN: {before} -> {len(df)}")

    return df

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna a lista de colunas que são features, excluindo colunas brutas e target
    """
    RAW_COLS = {"Date", "Open", "High", "Low", "Close", "Volume", "target", "ticker"}
    return [col for col in df.columns if col not in RAW_COLS]

def log_target_distribution(df: pd.DataFrame) -> None:
    """
    Imprime a distribuição do target para diferentes thresholds
    """
    logger.info("Distribuição do target por threshold:")
    for t in [0.01, 0.02, 0.03, 0.05]:
        pct = (
            df.groupby("ticker")["Close"]
            .transform(lambda x: x.shift(-TARGET_HORIZON) / x - 1) >= t
        ).mean()
        logger.info(f"  >= {t * 100:.0f%%}: {pct * 100:.1f%%} positivos")