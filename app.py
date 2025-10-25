
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    bernoulli, binom, poisson, randint, beta, norm, lognorm, expon, gamma, triang, uniform
)
from scipy.stats import skew, kurtosis

# =========================
# Funciones auxiliares
# =========================
def frequency_function(frequency_type, params, size):
    if frequency_type == 'bernoulli':
        return bernoulli.rvs(p=params['p'], size=size)
    elif frequency_type == 'binomial':
        return binom.rvs(n=params['n'], p=params['p'], size=size)
    elif frequency_type == 'poisson':
        return poisson.rvs(mu=params['mu'], size=size)
    elif frequency_type == 'uniform_int':
        return randint.rvs(low=params['low'], high=params['high'], size=size)
    else:
        raise ValueError("Unsupported frequency function")

# PERT (clásico) con lambda (shape)
def pert_rvs(a, b, c, lambd, size):
    mu = (a + lambd * b + c) / (lambd + 2)
    sigma = (c - a) / (lambd + 2) * np.sqrt(1 / (lambd + 3))
    alpha = ((mu - a) * (2 * c - a - mu)) / ((c - mu) * (mu - a))
    beta_ = alpha * (c - mu) / (mu - a)
    return beta.rvs(alpha, beta_, loc=a, scale=c - a, size=size)

# PERT definido por percentil del modo
def pert_percentile_rvs(a, b, c, p, size):
    alpha = 1 + p * (b - a) / (c - a)
    beta_ = 1 + p * (c - b) / (c - a)
    return beta.rvs(alpha, beta_, loc=a, scale=c - a, size=size)

# Triangular por percentil del modo
def triangular_perc_rvs(a, b, c, p, size):
    c_rel = (b - a) / (c - a)
    return triang.rvs(c_rel, loc=a, scale=c - a, size=size)

# =========================
# Severidad
# =========================
def severity_function(severity_type, params, size):
    if severity_type == 'normal':
        return norm.rvs(loc=params['loc'], scale=params['scale'], size=size)

    elif severity_type == 'lognormal':
        mu_usd = params['mu']
        sigma_usd = params['sigma']
        s_logn = np.sqrt(np.log(1 + (sigma_usd**2) / (mu_usd**2)))
        m_logn = np.log(mu_usd) - 0.5 * (s_logn**2)
        return lognorm.rvs(s=s_logn, scale=np.exp(m_logn), size=size)

    elif severity_type == 'lognormal_log':
        return lognorm.rvs(s=params['sigma_log'], scale=np.exp(params['mu_log']), size=size)

    elif severity_type == 'exponential':
        return expon.rvs(scale=params['scale'], size=size)

    elif severity_type == 'gamma':
        return gamma.rvs(a=params['a'], scale=params['scale'], size=size)

    elif severity_type == 'triangular':
        return triang.rvs(c=params['c'], loc=params['loc'], scale=params['scale'], size=size)

    elif severity_type == 'uniform':
        return uniform.rvs(loc=params['loc'], scale=params['scale'], size=size)

    elif severity_type == 'pert':
        return pert_rvs(params['a'], params['b'], params['c'], params['lambda'], size=size)

    elif severity_type == 'pert_percentile':
        return pert_percentile_rvs(params['a'], params['b'], params['c'], params['p'], size=size)

    elif severity_type == 'triangular_perc':
        return triangular_perc_rvs(params['a'], params['b'], params['c'], params['p'], size=size)

    else:
        raise ValueError("Unsupported severity function")

# =========================
# Simulador Monte Carlo
# =========================
def montecarlo_simulator(frequency_type, frequency_params, severity_type, severity_params, percentile, iterations):
    losses = []
    records = []

    for _ in range(iterations):
        frequency = int(frequency_function(frequency_type, frequency_params, 1)[0])
        if frequency > 0:
            sev = severity_function(severity_type, severity_params, frequency)
            total_loss = float(np.sum(sev))
        else:
            total_loss = 0.0

        losses.append(total_loss)
        records.append({'frequency': frequency})

    df = pd.DataFrame(records)
    df['loss'] = losses
    losses_array = df['loss'].to_numpy()

    stats = {
        'mean_loss': float(np.mean(losses_array)),
        'std_loss': float(np.std(losses_array, ddof=1)),
        'var_loss': float(np.percentile(losses_array, percentile * 100)),
        'unexpected_loss': float(np.percentile(losses_array, percentile * 100) - np.mean(losses_array)),
        'skewness': float(skew(losses_array)),
        'kurtosis': float(kurtosis(losses_array))
    }
    return df, stats

# =========================
# Interfaz Streamlit
# =========================
st.title("Simulador Montecarlo para Valuación de Riesgos")

st.sidebar.header("Horizonte del cálculo")
horizonte = st.sidebar.selectbox("Horizon", ["Mensual", "Trimestral", "Anual"])
mu_default = 0.35 if horizonte == "Mensual" else (1.05 if horizonte == "Trimestral" else 4.20)

st.sidebar.header("Frecuencia")
freq_type = st.sidebar.selectbox("Función de frecuencia", ["bernoulli", "binomial", "poisson", "uniform_int"])
freq_params = {}
if freq_type == 'poisson':
    freq_params['mu'] = st.sidebar.number_input(f"Tasa media de eventos ({horizonte.lower()})", min_value=0.0, value=mu_default)

st.sidebar.header("Severidad")
sev_type = st.sidebar.selectbox("Función de severidad", ["lognormal", "lognormal_log"])
sev_params = {}
if sev_type == 'lognormal':
    sev_params['mu'] = st.sidebar.number_input("Media (USD)", value=66228.62)
    sev_params['sigma'] = st.sidebar.number_input("Desv. estándar (USD)", value=89733.54)
elif sev_type == 'lognormal_log':
    sev_params['mu_log'] = st.sidebar.number_input("mu_log", value=10.5797)
    sev_params['sigma_log'] = st.sidebar.number_input("sigma_log", value=1.0209)

percentile = st.sidebar.slider("Percentil VaR", min_value=0.5, max_value=0.995, value=0.95)
iterations = st.sidebar.number_input("Iteraciones", min_value=100, value=1000)

if st.button("Ejecutar Simulación"):
    df, stats = montecarlo_simulator(freq_type, freq_params, sev_type, sev_params, percentile, iterations)

    st.subheader("Resultados estadísticos del modelo")
    st.write(f"Media: {stats['mean_loss']:,.2f}")
    st.write(f"Desv. estándar: {stats['std_loss']:,.2f}")
    st.write(f"VaR ({percentile*100:.1f}%): {stats['var_loss']:,.2f}")
    st.write(f"Pérdida no esperada: {stats['unexpected_loss']:,.2f}")
    st.write(f"Asimetría: {stats['skewness']:.4f}")
    st.write(f"Curtosis: {stats['kurtosis']:.4f}")
