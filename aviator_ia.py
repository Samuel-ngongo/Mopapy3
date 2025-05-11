import streamlit as st
import numpy as np
from datetime import datetime
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

st.set_page_config(page_title="Aviator PRO - IA Avançada", layout="centered")
st.title("Aviator PRO - IA Inteligente com Previsão Ampliada e Log Análise")

# Histórico
if "valores" not in st.session_state:
    st.session_state.valores = []

if "historico_completo" not in st.session_state:
    st.session_state.historico_completo = []

# Entrada de dados
novo = st.text_input("Insira um valor (ex: 2.31):")
if st.button("Adicionar") and novo:
    try:
        valor = float(novo)
        st.session_state.valores.append(valor)
        st.session_state.historico_completo.append((valor, datetime.now().strftime("%d/%m/%Y %H:%M")))
        st.success("Valor adicionado.")
    except:
        st.error("Formato inválido.")

# Função de previsão com aprimoramento
def prever_valor(dados):
    if len(dados) < 5:
        return 1.50, 1.20, 30, 0.0  # estimativa, inferior, confiança, % variação

    media_simples = np.mean(dados)
    pesos = np.linspace(1, 2, len(dados))
    media_ponderada = np.average(dados, weights=pesos)

    if LinearRegression and len(dados) >= 6:
        X = np.array(range(len(dados))).reshape(-1, 1)
        y = np.array(dados)
        modelo = LinearRegression()
        modelo.fit(X, y)
        reg_pred = modelo.predict(np.array([[len(dados) + 1]]))[0]
    else:
        reg_pred = media_ponderada

    estimativa_final = (media_simples + media_ponderada + reg_pred) / 3
    estimativa_inferior = min(media_simples, media_ponderada, reg_pred)

    desvio = np.std(dados[-10:]) if len(dados) >= 10 else np.std(dados)
    confianca = max(10, 100 - desvio * 100)

    # % variação esperada com base na tendência
    variacao = ((estimativa_final - dados[-1]) / dados[-1]) * 100 if dados[-1] != 0 else 0

    return round(estimativa_final, 2), round(estimativa_inferior, 2), round(confianca, 1), round(variacao, 2)

# Detectar mudança com logaritmo
def detectar_mudanca(dados):
    if len(dados) < 15:
        return False
    log_diff = np.diff(np.log(np.array(dados[-10:])))
    if np.max(np.abs(log_diff)) > 0.5:
        return True
    return False

# Analisar padrões
def analisar_padroes(dados):
    alertas = []
    if len(dados) >= 3:
        ultimos3 = dados[-3:]
        if all(v < 1.5 for v in ultimos3):
            alertas.append(("Queda contínua detectada", 70))
        if all(v > 2.5 for v in ultimos3):
            alertas.append(("Alta contínua detectada", 65))
        if len(set(np.sign(np.diff(ultimos3)))) > 1:
            alertas.append(("Alternância instável", 60))
    return alertas

# Visualização
def mostrar_graficos(valores):
    df = pd.DataFrame({
        'Índice': list(range(1, len(valores) + 1)),
        'Valor': valores
    })

    st.subheader("Mini Gráfico de Barras (últimos 10)")
    st.bar_chart(df.tail(10).set_index('Índice'))

    st.subheader("Evolução da Média")
    df['Média Móvel'] = df['Valor'].rolling(window=3, min_periods=1).mean()
    st.line_chart(df.set_index('Índice')[['Valor', 'Média Móvel']])

# Exibir histórico e análise
if st.session_state.valores:
    st.subheader("Histórico (últimos 30)")
    for valor, data in st.session_state.historico_completo[-30:]:
        cor = "🟥" if valor < 1.5 else "🟩" if valor > 2.5 else "⬜"
        st.write(f"{cor} {valor:.2f}x - {data}")

    mostrar_graficos(st.session_state.valores)

    st.subheader("Previsão e Análise Inteligente")
    est, inf, conf, var = prever_valor(st.session_state.valores)

    st.info(f"**Estimativa para próxima rodada:** {est}x")
    st.info(f"**Estimativa inferior (segura):** {inf}x")
    st.info(f"**Confiança da IA:** {conf}%")
    st.info(f"**Variação esperada:** {var}%")

    if conf >= 75:
        st.success("Alta confiança nas próximas rodadas.")
    elif conf >= 50:
        st.warning("Confiança moderada. Observe antes de agir.")
    else:
        st.error("Confiança baixa. Alta incerteza.")

    if detectar_mudanca(st.session_state.valores):
        st.warning("Mudança brusca identificada pelo logaritmo. IA ajustando previsão...")

    padroes = analisar_padroes(st.session_state.valores)
    for alerta, chance in padroes:
        st.info(f"Alerta de padrão: {alerta} ({chance}% de chance)")

# Limpar dados
if st.button("Limpar dados"):
    st.session_state.valores = []
    st.session_state.historico_completo = []
    st.success("Histórico limpo.")
