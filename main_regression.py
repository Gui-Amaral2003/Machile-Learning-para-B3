"""
================================================================================
STOCK SELECTION MODEL — PREVISÃO DE ALPHA EM 10 DIAS (REGRESSÃO)
================================================================================

OBJETIVO:
    Prever o "Alpha" (Retorno da Ação - Retorno do Ibovespa) em 10 pregões
    para ações brasileiras (mid/small caps). 

ESTRATÉGIA (V2 - Anti-Overfitting):
    1. Coleta: Dados de 2010 até hoje via Yahoo Finance, 23 tickers em 6 setores
    2. Features: Reduzidas para focar em indicadores-chave e diminuir colinearidade.
    3. Target: Alpha de 10 dias (Stock Return 10d - IBOV Return 10d).
    4. Modelos: XGBoost Regressor com hiperparâmetros rigorosamente restritivos
       (max_depth baixo, min_child_weight alto) para forçar generalização.
    5. Validação: PurgedTimeSeriesSplit. Análise final aplica um filtro de 
       Trend Following (só operar se o preço estiver acima da MM200).
================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from cv import PurgedTimeSeriesSplit
import warnings
from curl_cffi import requests
from datetime import datetime
from scipy.stats import pearsonr

session = requests.Session(impersonate='chrome')
warnings.filterwarnings('ignore')

## 1. COLETA DE DADOS
dfs = []
areas = {
    'MINERACAO E SIDERURGIA': ['VALE3.SA', 'CMIN3.SA', 'GGBR4.SA', 'USIM5.SA', 'CSNA3.SA'],
    'FINANCEIRAS': ['ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'BPAC11.SA', 'B3SA3.SA'],
    'PETROLEO, GAS E BIOCOMBUSTIVEIS': ['PETR4.SA', 'PETR3.SA', 'PRIO3.SA', 'UGPA3.SA'],
    'ALIMENTOS E BEBIDAS': ['BRFS3.SA', 'CRFB3.SA', 'JBSS3.SA'],
    'UTILIDADE PUBLICA': ['AXIA3.SA', 'SBSP3.SA'],
    'INDUSTRIA E CONSUMO': ['LREN3.SA', 'RADL3.SA', 'MGLU3.SA', 'RENT3.SA', 'EMBR3.SA', 'CBAV3.SA']
}

print("Coletando dados das ações...")
for area, tickers in areas.items():
    for ticker in tickers:
        temp_df = yf.download(
            ticker,
            start='2010-01-01',
            end=datetime.today().strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
            session=session
        )

        if temp_df.empty:
            continue

        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = temp_df.columns.get_level_values(0)

        temp_df = temp_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        temp_df['ticker'] = ticker.replace('.SA', '')
        temp_df['sector'] = area
        dfs.append(temp_df)

print("Coletando IBOVESPA...")
ibov = yf.download(
        '^BVSP', 
        start='2010-01-01', 
        end=datetime.today().strftime('%Y-%m-%d'), 
        auto_adjust=True, 
        progress=False, 
        session=session
    )

if isinstance(ibov.columns, pd.MultiIndex):
    ibov.columns = ibov.columns.get_level_values(0)

df = pd.concat(dfs).reset_index()

# Remover mega caps
mega_caps = ['PETR4', 'PETR3', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3']
df = df[~df['ticker'].isin(mega_caps)]


## 2. DEFINIÇÃO DE FEATURES (Enxutas)
print("Calculando features essenciais...")

# Apenas retornos de médio prazo para evitar colinearidade extrema de curtos prazos
for n in [5, 10, 20]:
    df[f'return_{n}'] = df.groupby('ticker')['Close'].transform(lambda x: x.pct_change(n))

# Distância para médias chave (reversão à média e tendência principal)
df['sma_ratio_20'] = df.groupby('ticker')['Close'].transform(lambda x: x / x.rolling(20).mean() - 1)
df['sma_ratio_50'] = df.groupby('ticker')['Close'].transform(lambda x: x / x.rolling(50).mean() - 1)

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta).where(delta < 0, 0).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))
 
df['rsi_14'] = df.groupby('ticker')['Close'].transform(lambda x: rsi(x, period=14))

# MACD
df['macd'] = df.groupby('ticker')['Close'].transform(lambda x: (x.ewm(span=12).mean() - x.ewm(span=26).mean()) / x)
df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9).mean())
df['macd_hist'] = df['macd'] - df['macd_signal']

# Volatilidade
df['vol_20'] = df.groupby('ticker')['return_5'].transform(lambda x: x.rolling(20).std()) # Usando volatilidade de retorno 5d para estabilizar
df['volatility_change'] = df.groupby('ticker')['vol_20'].transform(lambda x: x / x.rolling(20).mean() - 1)

# Filtro de Tendência Maior (usaremos na avaliação final)
sma200 = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(200).mean())
df['above_sma_200'] = (df['Close'] > sma200).astype(int)

# Preparando dados do Ibovespa para juntar
ibov_features = ibov[['Close']].copy()
ibov_features = ibov_features.rename(columns={'Close': 'ibov_close'})
ibov_features['ibov_ret_1d'] = ibov_features['ibov_close'].pct_change()
for n in [5, 10, 20]:
    ibov_features[f'ibov_ret_{n}'] = ibov_features['ibov_close'].pct_change(n)

df = df.join(ibov_features, on='Date')

# Alpha Histórico (Performance relativa passada)
df['alpha_5d'] = df['return_5'] - df['ibov_ret_5']
df['alpha_20d'] = df['return_20'] - df['ibov_ret_20']

# Z-Score do preço em relação a 20 dias
roll_mean_20 = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(20).mean())
roll_std_20  = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(20).std())
df['zscore_20']  = (df['Close'] - roll_mean_20) / roll_std_20

# Dummies de Setor
df = pd.get_dummies(df, columns=['sector'], prefix='sec', dtype=int)


## 3. VARIÁVEL ALVO (TARGET) - ALPHA DE 10 DIAS
# Quanto a ação rendeu em 10 dias MENOS quanto o Ibovespa rendeu em 10 dias.
# Valores > 0 indicam que a ação superou o mercado (gerou Alpha positivo).
ret_fut_10d = df.groupby('ticker')['Close'].transform(lambda x: x.shift(-10) / x - 1)
ibov_ret_fut_10d = df.groupby('ticker')['ibov_close'].transform(lambda x: x.shift(-10) / x - 1)

df['target'] = ret_fut_10d - ibov_ret_fut_10d


## 4. PREPARAÇÃO DOS DADOS
df = df.dropna()

# Remover as colunas que serviram só de base
RAW_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'ticker', 'ibov_close', 'ibov_ret_1d', 'ibov_ret_5', 'ibov_ret_10', 'ibov_ret_20']
features = [col for col in df.columns if col not in RAW_COLS]

print(f"\nFeatures utilizadas ({len(features)}): {', '.join(features)}")
print(f"Amostras totais: {len(df):,}")
print(f"Target Médio (Alpha 10d): {df['target'].mean():.4f} | Desvio Padrão: {df['target'].std():.4f}")


## 5. SPLIT TEMPORAL
cutoff = df['Date'].quantile(0.8)
print(f"\nCutoff de split: {pd.Timestamp(cutoff).date()}")
 
train_mask = df['Date'] <= cutoff
X_train = df.loc[train_mask,  features]
X_test  = df.loc[~train_mask, features]
y_train = df.loc[train_mask,  'target']
y_test  = df.loc[~train_mask, 'target']


## 6. NORMALIZAÇÃO
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def evaluate_regression(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    print(f"{prefix}RMSE : {rmse:.4f}")
    print(f"{prefix}MAE  : {mae:.4f}")
    print(f"{prefix}R2   : {r2:.4f}")
    print(f"{prefix}Corr : {corr:.4f}")
    return rmse, mae, r2, corr


## 7. TUNING XGBOOST (Regularização Extrema)
print("\n" + "=" * 60)
print("TUNING — XGBoost com Alta Regularização")
print("=" * 60)

# Espaço de busca muito mais restrito para forçar generalização
param_dist = {
    'max_depth': [2, 3, 4],            # Árvores bem rasas
    'min_child_weight': [20, 50, 100], # Exige clusters maiores de amostras
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 200, 300],
    'subsample': [0.5, 0.7, 0.8],
    'colsample_bytree': [0.5, 0.7],
    'reg_alpha': [0.1, 1.0, 5.0],      # Regularização L1 forte
    'reg_lambda': [1.0, 5.0, 10.0]     # Regularização L2 forte
}

xgb_base = XGBRegressor(
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)

cv = PurgedTimeSeriesSplit(gap_days=10) # Gap ajustado para o novo horizonte de 10 dias
search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=40,
    cv=cv,
    scoring='neg_mean_squared_error', 
    n_jobs=-1,
    verbose=1,
    random_state=42
)
search.fit(X_train_s, y_train, groups=df.loc[train_mask, 'Date'].values)

print(f"\nMelhores parâmetros encontrados:")
for k, v in search.best_params_.items():
    print(f"  {k:<22} = {v}")


## 8. MODELO FINAL (Tuned)
print("\n" + "=" * 60)
print("AVALIAÇÃO FINAL - XGBOOST")
print("=" * 60)

best_params = search.best_params_.copy()

xgb_tuned = XGBRegressor(
    **best_params,
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)

# Como já super-regularizamos na busca de hiperparâmetros, treinaremos no conjunto completo de treino
xgb_tuned.fit(X_train_s, y_train)

xgb_tuned_pred_train = xgb_tuned.predict(X_train_s)
xgb_tuned_pred_test  = xgb_tuned.predict(X_test_s)

print("\nResultados Treino:")
evaluate_regression(y_train, xgb_tuned_pred_train)
print("\nResultados Teste:")
xgb_t_rmse, xgb_t_mae, xgb_t_r2, xgb_t_corr = evaluate_regression(y_test, xgb_tuned_pred_test)


## 9. ANÁLISE DE DECIS E TREND FOLLOWING
print("\n" + "=" * 60)
print("ANÁLISE DE DECIS DE PREVISÃO")
print("=" * 60)

test_df = df.loc[~train_mask].copy()
test_df['pred_alpha'] = xgb_tuned_pred_test

def analise_decis(dataframe, titulo):
    if len(dataframe) == 0:
        return
        
    print(f"\n--- {titulo} ---")
    dataframe["decil"] = pd.qcut(
        dataframe["pred_alpha"],
        10,
        labels=False,
        duplicates="drop"
    )

    decil_analysis = dataframe.groupby("decil").agg(
        count=('target', 'count'),
        mean_real_alpha=('target', 'mean'),
        mean_pred_alpha=('pred_alpha', 'mean')
    ).sort_index(ascending=False) 
    
    print(decil_analysis.to_string())

# 1. Análise sem filtro
analise_decis(test_df, "TODOS OS DADOS DO TESTE")

# 2. Análise com Trend Following (Apenas operar se Preço > MM200)
# Queremos ver se as apostas de alta do modelo funcionam melhor quando a ação JÁ ESTÁ em alta
test_df_trend = test_df[test_df['above_sma_200'] == 1].copy()
analise_decis(test_df_trend, "COM FILTRO TREND FOLLOWING (Preço > MM200)")

print("\nConclusão Esperada: Com o filtro de tendência e a regularização extrema, espera-se")
print("que o Decil 9 da segunda tabela tenha um Alpha real consistentemente positivo.")
