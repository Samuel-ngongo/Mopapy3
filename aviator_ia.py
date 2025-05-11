import streamlit as st
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Aviator PRO - IA Adaptativa Total", layout="centered")
st.title("Aviator PRO - IA Inteligente com Padrões, Confiança e Histórico")

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

# Previsão com média ponderada + regressão
def prever_com_tecnicas(dados):
    if len(dados) < 5:
        return 1.50, 30
    # Média ponderada
    pesos = np.linspace(1, 2, len(dados))
    media_ponderada = np.average(dados, weights=pesos)

    # Regressão linear
    X = np.arange(len(dados)).reshape(-1, 1)
    y = np.array(dados)
    modelo = LinearRegression().fit(X, y)
    previsao_regressao = modelo.predict([[len(dados)]])[0]

    # Combinação (média simples entre as duas técnicas)
    combinada = (media_ponderada + previsao_regressao) / 2
    desvio = np.std(dados[-10:]) if len(dados) >= 10 else np.std(dados)
    confianca = max(10, 100 - desvio * 100)

    return round(combinada, 2), round(confianca, 1)

# Simulação de 3 a 5 rodadas futuras com base no padrão
def simular_futuro(dados, n=5):
    simulacoes = []
    for i in range(1, n + 1):
        if len(dados) < 5:
            simulacoes.append(1.50)
            continue
        dados_ficticios = dados + simulacoes
        estimativa, _ = prever_com_tecnicas(dados_ficticios)
        simulacoes.append(estimativa)
    return simulacoes

# Detectar mudança brusca
def detectar_mudanca(dados):
    if len(dados) < 15:
        return False
    ultimos = np.array(dados[-5:])
    anteriores = np.array(dados[-10:-5])
    media_diff = abs(np.mean(ultimos) - np.mean(anteriores))
    desvio_diff = abs(np.std(ultimos) - np.std(anteriores))
    return media_diff > 1.0 or desvio_diff > 1.2

# Analisar padrões de repetição
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

# Exibir histórico e análise
if st.session_state.valores:
    st.subheader("Histórico (últimos 30)")
    for valor, data in st.session_state.historico_completo[-30:]:
        st.write(f"{valor:.2f}x - {data}")

    st.subheader("Previsão e Análise Inteligente")
    estimativa, confianca = prever_com_tecnicas(st.session_state.valores)
    st.info(f"Estimativa combinada: {estimativa}x")
    st.info(f"Nível de confiança: {confianca}%")

    if confianca >= 75:
        st.success("Alta confiança nas próximas rodadas.")
    elif confianca >= 50:
        st.warning("Confiança moderada. Observe antes de agir.")
    else:
        st.error("Confiança baixa. Alta incerteza.")

    if detectar_mudanca(st.session_state.valores):
        st.warning("Mudança brusca de padrão detectada. IA recalibrando...")

    padroes = analisar_padroes(st.session_state.valores)
    for alerta, chance in padroes:
        st.info(f"Alerta de padrão: {alerta} ({chance}% de chance)")

    st.subheader("Simulação das Próximas Rodadas")
    simulacoes = simular_futuro(st.session_state.valores, 5)
    for i, s in enumerate(simulacoes, 1):
        st.write(f"Rodada +{i}: Estimativa {s}x")

# Limpar dados
if st.button("Limpar dados"):
    st.session_state.valores = []
    st.session_state.historico_completo = []
    st.success("Histórico limpo.")
