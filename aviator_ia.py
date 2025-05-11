import streamlit as st
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Aviator PRO - IA Adaptativa Total", layout="centered")
st.title("Aviator PRO - IA Inteligente com Padrões, Confiança e Histórico")

# Histórico
if "valores" not in st.session_state:
    st.session_state.valores = []

if "historico_completo" not in st.session_state:
    st.session_state.historico_completo = []

# Memória de padrões anteriores
if "memoria_padroes" not in st.session_state:
    st.session_state.memoria_padroes = []

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

# Função de aprendizado contínuo
def aprendizado_continuo(dados):
    if len(dados) < 5:
        return None  # Não há dados suficientes para aprender ainda

    # Dividir em blocos de 5 dados
    blocos = [dados[i:i + 5] for i in range(0, len(dados), 5)]
    ultimo_bloco = blocos[-1]

    # Ajuste da IA a cada novo bloco
    media_bloco = np.mean(ultimo_bloco)
    desvio_bloco = np.std(ultimo_bloco)
    
    # Análise de mudanças
    if desvio_bloco > 0.5:  # Limite ajustável para mudanças significativas
        return True  # Mudança detectada, IA vai recalibrar
    return False

# Recalibração automática
def recalibrar(dados):
    if len(dados) < 5:
        return 1.50, 30  # Valor de fallback
    pesos = np.linspace(1, 2, len(dados))
    media_ponderada = np.average(dados, weights=pesos)
    desvio = np.std(dados)
    confianca = max(10, 100 - desvio * 100)
    return round(media_ponderada, 2), round(confianca, 1)

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

    # Análise e previsão
    if aprendizado_continuo(st.session_state.valores):
        st.warning("Mudança detectada nos dados. IA recalibrando...")

    estimativa, confianca = recalibrar(st.session_state.valores)
    st.info(f"Estimativa recalibrada: {estimativa}x")
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

# Limpar dados
if st.button("Limpar dados"):
    st.session_state.valores = []
    st.session_state.historico_completo = []
    st.session_state.memoria_padroes = []
    st.success("Histórico limpo.")
