"""
================================================================================
STOCK SELECTION MODEL — PREVISÃO DE ALTA DE 5% EM 5 DIAS
================================================================================

OBJETIVO:
    Identificar ações brasileiras (mid/small caps) com alta probabilidade de
    valorização ≥ 5% nos próximos 5 dias, usando Machine Learning com features
    de análise técnica, momento e relação com o IBOVESPA.

MOTIVAÇÃO:
    - Mega caps (PETR4, VALE3, ITUB4, etc.) tendem a ter movimentos mais
      previsíveis e menos "explosivos", diluindo o sinal para estratégias
      de swing trade de curto prazo.
    - Mid/small caps oferecem mais oportunidades de movimentos significativos
      em janelas curtas, mas exigem features mais sofisticadas para capturar
      os padrões de forma consistente.

ESTRATÉGIA:
    1. Coleta: Dados de 2010 até hoje via Yahoo Finance, 23 tickers em 6 setores
    2. Features: +50 indicadores incluindo retornos, médias móveis, RSI, MACD,
       volatilidade, volume, z-score, distância de máximas/mínimas de 52 semanas,
       beta e alpha em relação ao IBOV, além de dummies setoriais.
    3. Target: 1 se Close[t+5] / Close[t] - 1 ≥ 5%, 0 caso contrário
    4. Modelos: Random Forest (baseline) e XGBoost com tuning via
       RandomizedSearchCV usando PurgedTimeSeriesSplit para evitar data leakage
    5. Validação: Split temporal 80/20, early stopping e análise por ano/ticker

MÉTRICAS DE AVALIAÇÃO:
    - F1-Score (balanceia precisão e recall, adequado para classes desbalanceadas)
    - Análise de calibração por threshold de probabilidade
    - Performance por decil de confiança do modelo
    - Walk-forward por ano e por ticker individual

DEPENDÊNCIAS:
    - yfinance: coleta de dados de mercado
    - pandas, numpy: manipulação de dados
    - scikit-learn: Random Forest, StandardScaler, RandomizedSearchCV
    - xgboost: modelo principal com early stopping
    - curl_cffi: sessão HTTP para burlar rate-limiting do Yahoo Finance

LIMITAÇÕES CONHECIDAS:
    - Dados fundamentalistas (P/L, ROE, etc.) não incluídos — foco puramente
      em price action e indicadores técnicos
    - Look-ahead bias potencial se features não forem cuidadosamente defasadas
    - Target de 5% em 5 dias é agressivo — classe positiva ~17.7% das amostras
    - Não considera custos de transação, slippage ou liquidez real

USO PRÁTICO:
    Este modelo serve como filtro inicial para gerar uma lista curta de ações
    com maior probabilidade de alta. NÃO é um sinal de compra/venda definitivo.
    Recomenda-se usar em conjunto com:
    - Confirmação de tendência em timeframes maiores
    - Gestão de risco rigorosa (stop loss, position sizing)
    - Análise fundamentalista complementar

RESULTADOS ESPERADOS:
    - F1-Score no teste (dados não vistos): tipicamente 0.35-0.55
    - Modelo bem calibrado mostra taxas de acerto crescentes nos decis superiores
    - Tickers com maior previsibilidade variam conforme regime de mercado


================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from cv import PurgedTimeSeriesSplit
import warnings
from curl_cffi import requests
from datetime import datetime
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

        # Remove MultiIndex do yfinance
        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = temp_df.columns.get_level_values(0)

        temp_df = temp_df[
            ['Open', 'High', 'Low', 'Close', 'Volume']
        ].copy()

        temp_df['ticker'] = ticker.replace('.SA', '')
        temp_df['sector'] = area

        dfs.append(temp_df)

ibov = yf.download(
        '^BVSP', 
        start = '2010-01-01', 
        end = datetime.today().strftime('%Y-%m-%d'), 
        auto_adjust=True, 
        progress=False, 
        session=session
    )

df = pd.concat(dfs)
df = df.reset_index()

# Remover mega caps, pois elas não possuem tantos sinais quanto as mid e small caps
mega_caps = [
    'PETR4',
    'PETR3',
    'VALE3',
    'ITUB4',
    'BBDC4',
    'BBAS3'
]

df = df[~df['ticker'].isin(mega_caps)]
print(df['ticker'].value_counts())

## 2. DEFINIÇÃO DE FEATURES
# Retornos passados (variação percentual do preço de fechamento em relação a n dias atrás)
for n in [1, 2, 3, 5, 10, 20]:
    df[f'return_{n}'] = df.groupby('ticker')['Close'].transform(
        lambda x: x.pct_change(n)
    )

# Médias moveis (% do preço de fechamento em relação a média movel de w dias)
for w in [5, 10, 20, 50]:
    df[f'sma_ratio_{w}'] = df.groupby('ticker')['Close'].transform(
        lambda x: x / x.rolling(w).mean() - 1
    )

# RSI (indice de força relativa)
def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta).where(delta < 0, 0).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))
 
df['rsi_14'] = df.groupby('ticker')['Close'].transform(
    lambda x: rsi(x, period=14)
)

# MACD (distancia entre a media móvel exponencial de 12 dias e a média móvel exponencial de 26 dias)
# MACD > signal -> tendencia de alta
# MACD < signal -> tendência de baixa
df['macd'] = df.groupby('ticker')['Close'].transform(
    lambda x: (x.ewm(span=12).mean() - x.ewm(span=26).mean()) / x
)
df['macd_signal'] = df.groupby('ticker')['macd'].transform(
    lambda x: x.ewm(span=9).mean()
)
df['macd_hist'] = df['macd'] - df['macd_signal']

# Volatilidade e volume
df['vol_20'] = df.groupby('ticker')['return_1'].transform(
    lambda x: x.rolling(20).std()
)
df['vol_ratio'] = df.groupby('ticker')['Volume'].transform(
    lambda x: x / x.rolling(20).mean()
)

df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
 
df['volatility_change'] = df.groupby('ticker')['vol_20'].transform(
    lambda x: x / x.rolling(20).mean() - 1
)
df['volume_spike'] = df.groupby('ticker')['Volume'].transform(
    lambda x: (x > x.rolling(20).mean() * 2).astype(int)
)

# Tendencias de curto x longo prazo
sma50  = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(50).mean())
sma200 = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(200).mean())
 
df['trend_50_200'] = (sma50 > sma200).astype(int)
df['above_sma_200']= (df['Close'] > sma200).astype(int)
 
df['momentum_20'] = df.groupby('ticker')['Close'].transform(
    lambda x: x / x.shift(20) - 1
)

# Relativo ao indice Ibov
if isinstance(ibov.columns, pd.MultiIndex):
    ibov.columns = ibov.columns.get_level_values(0)

ibov_ret = ibov['Close'].pct_change().rename('ibov_ret')
df = df.join(ibov_ret, on = 'Date')

# Quanto uma ação superou/ficou abaixo do indice Ibov 
df['alpha_1d'] = df['return_1'] - df['ibov_ret']
df['alpha_5d'] = df['return_5'] - df.groupby('ticker')['ibov_ret'].transform(lambda x: x.rolling(5).sum())
df['alpha_20d'] = df['return_20'] - df.groupby('ticker')['ibov_ret'].transform(lambda x: x.rolling(20).sum())

# Volatilidade relativa ao indice Ibov
ibov_vol = ibov_ret.rolling(20).std()
df['ibov_vol'] = df['Date'].map(ibov_vol)

df['beta_20'] = df.groupby('ticker').apply(
    lambda g: g['return_1'].rolling(20).cov(
        g['ibov_ret']
    ) / g['ibov_ret'].rolling(20).var()
).reset_index(level=0, drop = True)

# Extras
roll_mean_20 = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(20).mean())
roll_std_20  = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(20).std())
 
df['zscore_20']  = (df['Close'] - roll_mean_20) / roll_std_20
 
df['dist_max52'] = df.groupby('ticker')['Close'].transform(
    lambda x: x / x.rolling(252).max() - 1          
)
df['dist_min52'] = df.groupby('ticker')['Close'].transform(
    lambda x: x / x.rolling(252).min() - 1          
)
 
df['dow'] = pd.to_datetime(df['Date']).dt.dayofweek
 
# Setor como variável categórica
df = pd.get_dummies(df, columns=['sector'], prefix='sec', dtype=int)


## 3. Variavel alvo (target)
# 1 se o valor da ação subir 5% ou mais no próximos 5 dias, 0 caso contrario
# preço futuro / preço atual - 1 >= 0.05 -> target = 1

ret_fut = df.groupby('ticker')['Close'].transform(lambda x: x.shift(-5) / x - 1)
df['target'] = (ret_fut >= 0.05).astype(int)

print("\nDistribuição do target por threshold:")
for t in [0.01, 0.02, 0.03, 0.05]:
    pct = (df.groupby('ticker')['Close'].transform(lambda x: x.shift(-5)/x - 1) >= t).mean()
    print(f"  >= {t:.0%}: {pct:.1%} positivos")

## 4. PREPARAÇÃO DOS DADOS
# Remover linhas com valores nulos (devido ao cálculo de features)
df = df.dropna()

RAW_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'ticker']
features = [col for col in df.columns if col not in RAW_COLS]

print(f"\nFeatures: {len(features)}")
print(f"Amostras totais: {len(df):,}")
print(f"Target — Alta: {df['target'].mean():.1%} | Baixa: {1-df['target'].mean():.1%}")

## 5. Split temporal
cutoff = df['Date'].quantile(0.8)   # data que divide 80% das linhas
print(f"\nCutoff de split: {pd.Timestamp(cutoff).date()}")
 
train_mask = df['Date'] <= cutoff
X_train = df.loc[train_mask,  features]
X_test  = df.loc[~train_mask, features]
y_train = df.loc[train_mask,  'target']
y_test  = df.loc[~train_mask, 'target']
 
print(f"Treino: {len(X_train):,} amostras | Teste: {len(X_test):,} amostras")
print(f"Tickers no teste: {df.loc[~train_mask, 'ticker'].nunique()}")


## 6. Normalização dos dados
# Scaler fit no treino
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

## 7. Treinamento do baseline (Random Forest)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth = 5,
    min_samples_leaf = 20,
    class_weight='balanced',
    random_state = 42,
    n_jobs = -1
)
model.fit(X_train_s, y_train)

rf_train_acc = f1_score(y_train, model.predict(X_train_s), pos_label = 1)
rf_test_acc  = f1_score(y_test,  model.predict(X_test_s), pos_label = 1)
print(f"F1 Treino : {rf_train_acc:.4f}")
print(f"F1 Teste  : {rf_test_acc:.4f}")
print(f"Diferença : {abs(rf_train_acc - rf_test_acc):.4f}")
print(classification_report(y_test, model.predict(X_test_s), target_names=['Baixa','Alta']))

## 8. XGBOOST - sem tuning
print("=" * 60)
print("XGBOOST — parâmetros iniciais")
print("=" * 60)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pw = neg / pos
print(f"scale_pos_weight = {scale_pw:.2f}  (neg={neg}, pos={pos})")

xgb = XGBClassifier(
    n_estimators = 200,
    max_depth = 5,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.7,
    scale_pos_weight = scale_pw,
    eval_metric = 'logloss',
    random_state = 42,
    n_jobs = -1
)
xgb.fit(
    X_train_s,
    y_train,
    eval_set = [(X_test_s, y_test)],
    verbose = False
)

xgb_train_acc = f1_score(y_train, xgb.predict(X_train_s), pos_label = 1)
xgb_test_acc  = f1_score(y_test,  xgb.predict(X_test_s), pos_label = 1)
print(f"F1 Treino : {xgb_train_acc:.4f}")
print(f"F1 Teste  : {xgb_test_acc:.4f}")
print(f"Diferença : {abs(xgb_train_acc - xgb_test_acc):.4f}")
print(classification_report(y_test, xgb.predict(X_test_s), target_names=['Baixa','Alta']))

## 9. Tuning com timeseriessplit + randomizedsearchcv
print("=" * 60)
print("TUNING — XGBoost com TimeSeriesSplit")
print("=" * 60)

param_dist = {
    # Arquitetura da arvore
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 5, 10, 20],
    
    #Taxa de aprendizado x número de arvores
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [200, 300, 500],

    #Regularização por amostragem
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],

    #Regularização L1 e L2
    'reg_alpha': [0, 0.01, 0.1, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0]
}

xgb_base = XGBClassifier(
    scale_pos_weight = scale_pw,
    eval_metric = 'logloss',
    random_state = 42,
    n_jobs = -1
)

cv = PurgedTimeSeriesSplit()
search = RandomizedSearchCV(
    estimator = xgb_base,
    param_distributions=param_dist,
    n_iter=40,
    cv=cv,
    scoring='f1',
    n_jobs = -1,
    verbose=1,
    random_state=42
)
search.fit(X_train_s, y_train, groups = df.loc[train_mask, 'Date'].values)

print("=== DIAGNÓSTICO DO PURGED CV ===")
dates_train = pd.to_datetime(df.loc[train_mask, 'Date'].values)

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_s, groups=dates_train)):
    tr_dates  = dates_train[tr_idx]
    val_dates = dates_train[val_idx]
    overlap   = val_dates.min() < tr_dates.max()
    print(f"Fold {fold+1}: treino até {tr_dates.max().date()} | "
          f"val {val_dates.min().date()}→{val_dates.max().date()} | "
          f"vazamento: {'⚠ SIM' if overlap else 'OK'}")

print(f"\nMelhores parâmetros encontrados:")
for k, v in search.best_params_.items():
    print(f"  {k:<22} = {v}")
print(f"\nMelhor F1 na CV: {search.best_score_:.4f}")

## 10. Modelo final com os melhores parametros
print("\n" + "=" * 60)
print("XGBOOST TUNADO — avaliação final no teste")
print("=" * 60)
 
best_params = search.best_params_.copy()
best_params['n_estimators'] = 1000
xgb_tuned = XGBClassifier(
    **best_params,
    scale_pos_weight = scale_pw,
    eval_metric = 'logloss',
    random_state = 42,
    n_jobs = -1,
    early_stopping_rounds = 30
)

# Usa 10% do treino como validação interna para o early stopping, evitando o overfitting
val_size   = int(len(X_train_s) * 0.1)
X_tr2      = X_train_s[:-val_size]
y_tr2      = y_train.iloc[:-val_size]
X_val2     = X_train_s[-val_size:]
y_val2     = y_train.iloc[-val_size:]

xgb_tuned.fit(
    X_tr2,
    y_tr2,
    eval_set = [(X_val2, y_val2)],
    verbose = False
)

print(f"Árvores usadas: {xgb_tuned.best_iteration} de 1000")

xgb_tuned_train = f1_score(y_train, xgb_tuned.predict(X_train_s), pos_label = 1)
xgb_tuned_test  = f1_score(y_test,  xgb_tuned.predict(X_test_s), pos_label = 1)
 
print(f"F1 Treino : {xgb_tuned_train:.4f}")
print(f"F1 Teste  : {xgb_tuned_test:.4f}")
print(f"Diferença : {abs(xgb_tuned_train - xgb_tuned_test):.4f}")
print(classification_report(y_test, xgb_tuned.predict(X_test_s), target_names=['Baixa','Alta']))

##11. Comparativo final
print("=" * 60)
print("RESUMO COMPARATIVO")
print("=" * 60)
print(f"{'Modelo':<25} {'Treino':>8} {'Teste':>8} {'Gap':>8}")
print("-" * 52)
print(f"{'Random Forest':<25} {rf_train_acc:>8.4f} {rf_test_acc:>8.4f} {abs(rf_train_acc-rf_test_acc):>8.4f}")
print(f"{'XGBoost (sem tuning)':<25} {xgb_train_acc:>8.4f} {xgb_test_acc:>8.4f} {abs(xgb_train_acc-xgb_test_acc):>8.4f}")
print(f"{'XGBoost (tunado)':<25} {xgb_tuned_train:>8.4f} {xgb_tuned_test:>8.4f} {abs(xgb_tuned_train-xgb_tuned_test):>8.4f}")

##12. Importância das features
print("\n" + "=" * 60)
print("TOP 10 FEATURES — XGBoost tunado")
print("=" * 60)
 
importances = pd.Series(
    xgb_tuned.feature_importances_,
    index=features
).sort_values(ascending=False)
 
print(importances.head(10).to_string())
print("\nFeatures com importância < 1% (candidatas a remover):")
print(importances[importances < 0.01].index.tolist())

from sklearn.metrics import f1_score

print("=== WALK-FORWARD POR ANO (teste real) ===")
test_df = df.loc[~train_mask].copy()
test_df['pred'] = xgb_tuned.predict(X_test_s)

# F1 por ano — revela em quais períodos o modelo funciona
for year in sorted(test_df['Date'].dt.year.unique()):
    mask_yr = test_df['Date'].dt.year == year
    if mask_yr.sum() < 100:
        continue
    f1 = f1_score(test_df.loc[mask_yr, 'target'],
                  test_df.loc[mask_yr, 'pred'])
    n  = mask_yr.sum()
    print(f"  {year}: F1 {f1:.4f}  ({n:,} amostras)")

# F1 por ticker — revela quais ações têm mais sinal
print("\n=== F1 POR TICKER ===")
results = []
for ticker in test_df['ticker'].unique():
    mask_tk = test_df['ticker'] == ticker
    if mask_tk.sum() < 50:
        continue
    f1 = f1_score(test_df.loc[mask_tk, 'target'],
                  test_df.loc[mask_tk, 'pred'])
    results.append({'ticker': ticker, 'f1': f1, 'n': mask_tk.sum()})

result = pd.DataFrame(results).sort_values('f1', ascending=False).to_string(index=False)
print(f"{result}\n")

test_df["proba"] = xgb_tuned.predict_proba(X_test_s)[:, 1]

for p in [0.5, 0.6, 0.7, 0.8]:
    mask = test_df["proba"] >= p

    print(
        p,
        len(test_df[mask]),
        test_df.loc[mask, "target"].mean()
    )

test_df["decil"] = pd.qcut(
    test_df["proba"],
    10,
    labels=False,
    duplicates="drop"
)

print(
    test_df.groupby("decil")["target"]
    .agg(["count","mean"])
)