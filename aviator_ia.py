import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Previsão de Queda - Aviator", layout="wide")
st.title("Previsão de Queda - Aviator")

# Estrutura para armazenar os dados
if 'valores' not in st.session_state:
    st.session_state.valores = []
if 'historico_completo' not in st.session_state:
    st.session_state.historico_completo = []

# Função para salvar os dados em um arquivo .txt
def salvar_historico():
    with open('historico_previsao.txt', 'a') as f:
        for valor in st.session_state.historico_completo:
            f.write(f"{valor['data']} - {valor['valor']}\n")

# Adicionar valor
valor = st.text_input("Digite um valor:")
if st.button("Adicionar"):
    if valor:
        valor_float = float(valor)
        st.session_state.valores.append(valor_float)
        # Adiciona ao histórico com data
        data_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.historico_completo.append({'data': data_hora, 'valor': valor_float})
        salvar_historico()
        st.success(f"Valor {valor_float} adicionado com sucesso!")

# Previsão e Lógica de IA Híbrida
def media_simples(valores):
    return np.mean(valores)

def media_ponderada(valores):
    pesos = np.arange(1, len(valores) + 1)
    return np.average(valores, weights=pesos)

def regressao_linear(valores):
    if len(valores) < 2:
        return None
    X = np.array(range(len(valores))).reshape(-1, 1)
    y = np.array(valores)
    modelo = LinearRegression()
    modelo.fit(X, y)
    previsao = modelo.predict(np.array([[len(valores)]]))
    return previsao[0]

def detectar_mudanca(valores):
    if len(valores) < 2:
        return None
    diff = valores[-1] - valores[-2]
    if abs(diff) > 1.5:  # Limite de mudança brusca
        return "Mudança brusca detectada"
    return "Sem mudança significativa"

def analisar_padroes(valores):
    if len(valores) >= 3:
        if all(v < 1.5 for v in valores[-3:]):
            return "Queda contínua"
        elif all(v > 2.5 for v in valores[-3:]):
            return "Alta contínua"
        elif (valores[-1] > valores[-2] and valores[-2] < valores[-3]) or (valores[-1] < valores[-2] and valores[-2] > valores[-3]):
            return "Alternância instável"
    return "Sem padrão detectado"

# Calculando as previsões
if len(st.session_state.valores) > 1:
    previsao_media = media_simples(st.session_state.valores)
    previsao_ponderada = media_ponderada(st.session_state.valores)
    previsao_linear = regressao_linear(st.session_state.valores)

    # Detectando mudanças
    mudanca = detectar_mudanca(st.session_state.valores)

    # Analisando padrões
    padrao = analisar_padroes(st.session_state.valores)

    st.subheader("Previsões")
    st.write(f"Média Simples: {previsao_media:.2f}")
    st.write(f"Média Ponderada: {previsao_ponderada:.2f}")
    st.write(f"Previsão Linear: {previsao_linear:.2f}")
    st.write(f"Detecção de Mudança: {mudanca}")
    st.write(f"Análise de Padrão: {padrao}")

# Exibindo gráficos
st.subheader("Evolução dos Valores")
if len(st.session_state.valores) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(st.session_state.valores, label="Valores", marker='o')
    ax.set_xlabel('Index')
    ax.set_ylabel('Valor')
    ax.set_title('Evolução dos Valores')
    ax.legend()
    st.pyplot(fig)

    # Gráfico de Média Móvel
    media_movel = pd.Series(st.session_state.valores).rolling(window=3).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(media_movel, label="Média Móvel", color='orange', linestyle='--')
    ax.set_xlabel('Index')
    ax.set_ylabel('Valor')
    ax.set_title('Média Móvel dos Últimos Valores')
    ax.legend()
    st.pyplot(fig)

# Exibindo histórico
st.subheader("Histórico Completo")
historico_df = pd.DataFrame(st.session_state.historico_completo)
st.write(historico_df)

# Limpar Dados
if st.button("Limpar Dados"):
    st.session_state.valores = []
    st.session_state.historico_completo = []
    st.success("Dados limpos com sucesso!")
