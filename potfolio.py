import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Configuración inicial
np.random.seed(152)

# --- 1. CONFIGURACIÓN DE ACTIVOS ---
# Hemos cambiado AAPL (Bono manual) por TLT (Bonos del Tesoro) y añadido GLD (Oro)
tickers = [
    "RSP",      # S&P 500 Equal Weight
    "XLV",      # Salud
    "QQQ",      # Nasdaq 100
    "NOBL",     # Dividend Aristocrats
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "URA",      # Uranio
    "VCSH",     # Bonos Corp. Corto Plazo
    "TLT",      # Bonos Tesoro 20+ Años (Sustituto del bono manual)
    "GLD"       # Oro
]

print(f"Descargando datos para: {tickers} ...")
# 2. Fechas exactas para tu prueba (Formato AÑO-MES-DÍA)
FECHA_INICIO = "2024-01-18"  # Fecha desde donde quieres historia
FECHA_FIN    = "2026-01-18"  # Hasta donde quieres ver qué pasó

print(f"📥 Descargando datos para: {tickers}")
print(f"📅 Desde {FECHA_INICIO} hasta {FECHA_FIN}...")

# Descarga usando fechas fijas (start/end) en lugar de periodo relativo
raw_data = yf.download(tickers, start=FECHA_INICIO, end=FECHA_FIN, progress=False)

# --- 2. LIMPIEZA DE DATOS ---
# Manejo robusto de multi-índices de Yahoo Finance
if isinstance(raw_data.columns, pd.MultiIndex):
    # Intentamos buscar 'Adj Close', si no 'Close'
    if 'Adj Close' in raw_data.columns.levels[0]:
        df = raw_data['Adj Close']
    elif 'Close' in raw_data.columns.levels[0]:
        df = raw_data['Close']
    else:
        # Fallback genérico
        df = raw_data.xs('Close', axis=1, level=0) 
else:
    df = raw_data[['Adj Close']] if 'Adj Close' in raw_data.columns else raw_data[['Close']]

# Asegurar que las columnas son solo los tickers
df.columns = [c for c in df.columns]

# Rellenar datos faltantes (por si algún activo tiene días festivos diferentes)
df = df.ffill().dropna()

print("Datos cargados correctamente.")
print("Activos procesados:", df.columns.tolist())

# --- 3. PARÁMETROS ESTADÍSTICOS (Todo automático de YFinance) ---
log_returns = np.log(df / df.shift(1)).dropna()

# Anualizamos media y desviación
mu_final = log_returns.mean() * 252
sigma_final = log_returns.std() * np.sqrt(252)
corr_matrix = log_returns.corr()

# --- 4. SIMULACIÓN MONTE CARLO (Cholesky) ---
diag_sigma = np.diag(sigma_final.values)
full_cov_matrix = diag_sigma @ corr_matrix.values @ diag_sigma

# Cholesky Decomposition
# Añadimos una minúscula regularización por si la matriz no es perfectamente definida positiva
try:
    L = np.linalg.cholesky(full_cov_matrix)
except np.linalg.LinAlgError:
    print("Ajustando matriz para Cholesky...")
    L = np.linalg.cholesky(full_cov_matrix + np.eye(len(full_cov_matrix)) * 1e-6)

n_simulaciones_base = 10000
n_assets = len(mu_final)
dt = 0.5 # 6 meses (Horizonte de proyección)

# Generación de ruido correlacionado
Z_indep = np.random.standard_normal((n_simulaciones_base, n_assets))
Z_corr = Z_indep @ L.T 

# Movimiento Browniano Geométrico
drift = (mu_final.values - 0.5 * sigma_final.values**2) * dt
retornos_simulados = np.exp(drift + Z_corr) - 1 
returns_df = pd.DataFrame(retornos_simulados, columns=mu_final.index)

# --- 5. CÁLCULO DE ESCENARIOS TIPO 1 (5 Ramas - Inicial) ---
print("\nCalculando escenarios iniciales (5 ramas)...")

# Ordenamos simulaciones según el retorno promedio de la cartera (Equiponderada para el orden)
cartera_avg = returns_df.mean(axis=1)
df_sorted = returns_df.loc[cartera_avg.sort_values().index]

# Definición de cortes de probabilidad
idx_10 = int(n_simulaciones_base * 0.10)
idx_30 = int(n_simulaciones_base * 0.30)
idx_70 = int(n_simulaciones_base * 0.70)
idx_90 = int(n_simulaciones_base * 0.90)

escenarios_finales = {}
escenarios_finales['Muy Bajista'] = df_sorted.iloc[0:idx_10].mean()      # 0-10%
escenarios_finales['Bajista']     = df_sorted.iloc[idx_10:idx_30].mean() # 10-30%
escenarios_finales['Neutral']     = df_sorted.iloc[idx_30:idx_70].mean() # 30-70%
escenarios_finales['Alcista']     = df_sorted.iloc[idx_70:idx_90].mean() # 70-90%
escenarios_finales['Muy Alcista'] = df_sorted.iloc[idx_90:].mean()       # 90-100%

df_escenarios_semestrales = pd.DataFrame(escenarios_finales)

print("\n--- TABLA DE ESCENARIOS INICIALES (5 Ramas) ---")
print(df_escenarios_semestrales.map(lambda x: f"{x:.2%}"))

# Guardar CSV 1
df_escenarios_semestrales.to_csv("escenarios_semestrales.csv")
print("✅ Guardado: escenarios_semestrales.csv")


# --- 6. CÁLCULO DE ESCENARIOS TIPO 2 (2 Ramas - Futuros) ---
print("\nCalculando escenarios futuros (2 ramas binarias)...")

mid_point = int(n_simulaciones_base * 0.50)

escenarios_futuros = {}
# Bajista Futuro: Promedio de la mitad inferior
escenarios_futuros['Bajista_Futuro'] = df_sorted.iloc[0:mid_point].mean()
# Alcista Futuro: Promedio de la mitad superior
escenarios_futuros['Alcista_Futuro'] = df_sorted.iloc[mid_point:].mean()

df_futuros = pd.DataFrame(escenarios_futuros)

print("\n--- TABLA DE ESCENARIOS FUTUROS (2 Ramas) ---")
print(df_futuros.map(lambda x: f"{x:.2%}"))

# Guardar CSV 2
df_futuros.to_csv("escenarios_futuros_binarios.csv")
print("✅ Guardado: escenarios_futuros_binarios.csv")


# --- 7. VISUALIZACIÓN ---
# Calculamos filas necesarias para los gráficos dinámicamente
n_plots = len(mu_final.index)
cols = 3
rows = math.ceil(n_plots / cols)

plt.figure(figsize=(15, 4 * rows))
plt.suptitle(f"Distribución Monte Carlo y Escenarios ({dt*12:.0f} meses)", fontsize=16)

branch_names = ["Muy Bajista", "Bajista", "Neutral", "Alcista", "Muy Alcista"]

for i, ticker in enumerate(mu_final.index):
    plt.subplot(rows, cols, i + 1)
    # Histograma de simulaciones
    sns.histplot(returns_df[ticker], kde=True, color='skyblue', stat="density", bins=50)
    plt.axvline(0, color='red', linestyle='--', alpha=0.6)
    
    # Pintamos las 5 líneas de los escenarios
    for b_name in branch_names:
        val = df_escenarios_semestrales.loc[ticker, b_name]
        plt.axvline(val, color='green', linestyle=':', alpha=0.9)
    
    plt.title(f"{ticker}")
    plt.xlabel("Retorno")
    plt.ylabel("Densidad")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n¡PROCESO COMPLETADO! Archivos listos para Julia.")