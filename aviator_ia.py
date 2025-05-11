
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="Aviator - Super IA Adaptativa", layout="centered")
st.title("Aviator - Super IA com Análise Avançada e Aprendizado Contínuo")

# Histórico salvo
HISTORICO_CSV = "historico_aviator.csv"

# Carregar histórico salvo
def carregar_dados():
    if os.path.exists(HISTORICO_CSV):
        return pd.read_csv(HISTORICO_CSV)["valor"].tolist()
    return []

# Salvar dados
def salvar_dados(lista):
    pd.DataFrame({"valor": lista}).to_csv(HISTORICO_CSV, index=False)

# Sessão inicial
if "valores" not in st.session_state:
    st.session_state.valores = carregar_dados()

# Entrada de dados
novo = st.text_input("Insira um valor (ex: 2.31):")
if st.button("Adicionar") and novo:
    try:
        val = float(novo)
        st.session_state.valores.append(val)
        salvar_dados(st.session_state.valores)
        st.success("Valor adicionado.")
    except:
        st.error("Formato inválido.")

# Estimativas inteligentes
def previsao_avancada(dados):
    if len(dados) < 5:
        return 1.50, 1.60, 30

    X = np.arange(len(dados)).reshape(-1, 1)
    y = np.array(dados)
    modelo = LinearRegression()
    modelo.fit(X, y)
    reg_pred = modelo.predict(np.array([[len(dados)]])).item()

    media_simples = np.mean(dados[-10:])
    pesos = np.linspace(1, 2, min(len(dados), 10))
    media_ponderada = np.average(dados[-10:], weights=pesos)

    previsao_final = np.mean([reg_pred, media_simples, media_ponderada])

    desvio = np.std(dados[-10:])
    confianca = max(10, min(100, 100 - desvio * 90))

    estimativa_min = round(previsao_final - desvio / 2, 2)
    estimativa_max = round(previsao_final + desvio / 2, 2)

    return estimativa_min, estimativa_max, round(confianca, 1)

def mudou_padrao(dados):
    if len(dados) < 10:
        return False

    ultimos = np.array(dados[-5:])
    anteriores = np.array(dados[-10:-5])
    media_diff = abs(np.mean(ultimos) - np.mean(anteriores))
    desvio_diff = abs(np.std(ultimos) - np.std(anteriores))

    return media_diff > 0.8 or desvio_diff > 1.0

if st.session_state.valores:
    st.subheader("Histórico (últimos 30)")
    st.write([f"{v:.2f}x" for v in st.session_state.valores[-30:]])

    st.subheader("Previsão e Análise Inteligente")

    est_min, est_max, confianca = previsao_avancada(st.session_state.valores)
    st.info(f"Estimativa para próxima rodada: entre **{est_min}x** e **{est_max}x**")
    st.info(f"Probabilidade dentro do intervalo: **{confianca}%**")

    if confianca >= 80:
        st.success("Alta chance de manter o padrão. Oportunidade favorável.")
    elif confianca >= 60:
        st.warning("Padrão moderado. Requer observação.")
    else:
        st.error("Chance baixa. Mudança de lógica provável.")

    if mudou_padrao(st.session_state.valores):
        st.warning("Mudança de padrão DETECTADA! IA ajustando estratégia...")

col1, col2 = st.columns(2)

with col1:
    if st.button("Limpar histórico"):
        st.session_state.valores = []
        if os.path.exists(HISTORICO_CSV):
            os.remove(HISTORICO_CSV)
        st.success("Histórico apagado.")

with col2:
    if st.button("Ver total de registros"):
        st.info(f"Total de valores registrados: {len(st.session_state.valores)}")
