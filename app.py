
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, binom, beta, lognorm, gamma

# -----------------------------
# Utility functions
# -----------------------------

def lognormal_params_from_mean_std(mean, std):
    if mean <= 0 or std <= 0:
        raise ValueError("Mean and std for lognormal must be positive.")
    variance = std ** 2
    sigma2 = np.log(1.0 + variance / (mean ** 2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    return mu, sigma

def gamma_params_from_mean_std(mean, std):
    if mean <= 0 or std <= 0:
        raise ValueError("Mean and std for gamma must be positive.")
    shape_k = (mean / std) ** 2
    scale_theta = (std ** 2) / mean
    return shape_k, scale_theta

def pert_sample(a, m, b, size):
    # (a) min, (m) most likely, (b) max
    if not (a < m < b):
        raise ValueError("Require a < m < b for PERT.")
    alpha = 1 + 4 * (m - a) / (b - a)
    beta_param = 1 + 4 * (b - m) / (b - a)
    u = np.random.beta(alpha, beta_param, size=size)
    return a + u * (b - a)

def simulate_losses(
    sims,
    horizon_periods,
    freq_model,
    lambda_per_period,
    p_event,
    severity_model,
    sev_params,
    random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Frequency
    if freq_model == "Poisson (múltiples eventos por periodo)":
        lam = max(lambda_per_period, 0.0) * max(horizon_periods, 0)
        N = np.random.poisson(lam=lam, size=sims)
    elif freq_model == "Binomial (0/1 evento por periodo)":
        n = max(int(horizon_periods), 0)
        p = min(max(p_event, 0.0), 1.0)
        N = np.random.binomial(n=n, p=p, size=sims)
    else:
        raise ValueError("Modelo de frecuencia no reconocido.")

    # Severidad
    losses = np.zeros(sims)
    if severity_model == "Lognormal (ingresa media y desviación)":
        mean, std = sev_params["mean"], sev_params["std"]
        mu, sigma = lognormal_params_from_mean_std(mean, std)
        # scipy's lognorm takes s (sigma) and scale=exp(mu)
        for i, n in enumerate(N):
            if n > 0:
                losses[i] = lognorm.rvs(s=sigma, scale=np.exp(mu), size=int(n)).sum()
    elif severity_model == "Gamma (ingresa media y desviación)":
        mean, std = sev_params["mean"], sev_params["std"]
        k, theta = gamma_params_from_mean_std(mean, std)
        for i, n in enumerate(N):
            if n > 0:
                losses[i] = gamma.rvs(a=k, scale=theta, size=int(n)).sum()
    elif severity_model == "PERT (mín, más probable, máx)":
        a, m, b = sev_params["a"], sev_params["m"], sev_params["b"]
        for i, n in enumerate(N):
            if n > 0:
                losses[i] = pert_sample(a, m, b, size=int(n)).sum()
    else:
        raise ValueError("Modelo de severidad no reconocido.")

    return losses, N

def var_es(losses, alpha=0.99):
    q = np.quantile(losses, alpha)
    es = losses[losses >= q].mean() if (losses >= q).any() else q
    return q, es

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Liquidity VaR - Funding Risk", page_icon="💧", layout="wide")
st.title("💧 Liquidity VaR (Funding Risk)")

with st.sidebar:
    st.header("Parámetros generales")
    sims = st.number_input("N° simulaciones", min_value=1000, max_value=200000, value=10000, step=1000)
    horizon_periods = st.number_input("Horizonte (n° de periodos, p.ej., meses)", min_value=1, max_value=120, value=1, step=1)
    alpha = st.slider("Nivel de confianza (VaR)", min_value=0.80, max_value=0.999, value=0.99, step=0.001)
    seed = st.number_input("Semilla aleatoria (opcional)", min_value=0, max_value=10_000_000, value=0, step=1)
    if seed == 0:
        seed = None

st.subheader("1) Frecuencia de eventos de iliquidez")
col1, col2 = st.columns(2)
with col1:
    freq_model = st.selectbox("Modelo de frecuencia", ["Poisson (múltiples eventos por periodo)", "Binomial (0/1 evento por periodo)"])
with col2:
    if freq_model.startswith("Poisson"):
        lambda_per_period = st.number_input("λ por periodo (promedio de eventos)", min_value=0.0, value=0.44, step=0.01, format="%.4f")
        p_event = 0.0
    else:
        p_event = st.number_input("Probabilidad de evento por periodo (p)", min_value=0.0, max_value=1.0, value=0.438, step=0.001, format="%.3f")
        lambda_per_period = 0.0

st.divider()
st.subheader("2) Severidad por evento (pérdida en $)")
sev_model = st.selectbox("Modelo de severidad", ["Lognormal (ingresa media y desviación)", "Gamma (ingresa media y desviación)", "PERT (mín, más probable, máx)"])

sev_params = {}
if sev_model in ["Lognormal (ingresa media y desviación)", "Gamma (ingresa media y desviación)"]:
    c1, c2 = st.columns(2)
    with c1:
        sev_params["mean"] = st.number_input("Media histórica ($)", min_value=0.0, value=66228.62, step=100.0, format="%.2f")
    with c2:
        sev_params["std"] = st.number_input("Desviación estándar ($)", min_value=0.0, value=89733.54, step=100.0, format="%.2f")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        sev_params["a"] = st.number_input("Mínimo ($)", min_value=0.0, value=14002.0, step=100.0, format="%.2f")
    with c2:
        sev_params["m"] = st.number_input("Más probable ($)", min_value=0.0, value=66228.62, step=100.0, format="%.2f")
    with c3:
        sev_params["b"] = st.number_input("Máximo ($)", min_value=0.0, value=239252.0, step=100.0, format="%.2f")

st.divider()
if st.button("Correr simulación"):
    try:
        losses, N = simulate_losses(
            sims=sims,
            horizon_periods=horizon_periods,
            freq_model=freq_model,
            lambda_per_period=lambda_per_period,
            p_event=p_event,
            severity_model=sev_model,
            sev_params=sev_params,
            random_seed=seed,
        )

        df = pd.DataFrame({"pérdida_total": losses, "n_eventos": N})
        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses, ddof=1))
        var_val, es_val = var_es(losses, alpha=float(alpha))

        st.write("### Resultados")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pérdida esperada (EL)", f"${mean_loss:,.0f}")
        m2.metric(f"VaR {int(alpha*100)}%", f"${var_val:,.0f}")
        m3.metric("Expected Shortfall (ES)", f"${es_val:,.0f}")
        m4.metric("Eventos esperados", f"{np.mean(N):.2f}")

        # Tabla resumida
        st.write("#### Percentiles de pérdida")
        percentiles = [0.50, 0.90, 0.95, 0.99, 0.995]
        percs = np.quantile(losses, percentiles)
        table = pd.DataFrame({"Percentil": [int(p*1000)/10 for p in percentiles], "Pérdida ($)": [f"{x:,.0f}" for x in percs]})
        st.dataframe(table, use_container_width=True)

        # Histograma (una sola figura, sin especificar colores)
        fig, ax = plt.subplots()
        ax.hist(losses, bins=60)
        ax.set_title("Distribución simulada de pérdidas")
        ax.set_xlabel("Pérdida total (por horizonte)")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

        # Descargar resultados
        st.download_button(
            label="Descargar pérdidas simuladas (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="liquidity_var_losses.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error en la simulación: {e}")
else:
    st.info("Configura los parámetros y pulsa **Correr simulación**.")
