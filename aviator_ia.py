import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Inicialização da sessão
if 'dados' not in st.session_state:
    st.session_state.dados = []

st.set_page_config(page_title="Aviator PRO IA", layout="centered")

# Estilo personalizado
st.markdown("""
    <style>
        .big-font { font-size:24px; font-weight:bold; color:#DAA520; }
        .emoji { font-size:30px; }
    </style>
""", unsafe_allow_html=True)

# Cabeçalho
st.markdown('<div class="big-font">✈️ Aviator PRO IA - Previsão de Quedas</div>', unsafe_allow_html=True)
st.write("Registe os valores das rodadas, visualize padrões e receba previsões com base em inteligência de dados.")

# Entrada de nova rodada
with st.form("entrada_dados"):
    nova_rodada = st.number_input("Digite o valor da rodada:", min_value=0.01, step=0.01, format="%.2f")
    enviar = st.form_submit_button("Registrar")

# Adicionar dados
if enviar and nova_rodada:
    st.session_state.dados.append({
        "Data/Hora": datetime.now().strftime("%d/%m %H:%M:%S"),
        "Valor": nova_rodada
    })

# Função para classificar valores
def classificar(valor):
    if valor < 2:
        return "Baixo", "⚠️"
    elif valor < 10:
        return "Médio", "✅"
    else:
        return "Alto", "🔥"

# Mostrar histórico
if st.session_state.dados:
    df = pd.DataFrame(st.session_state.dados)
    df["Classificação"], df["Emoji"] = zip(*df["Valor"].apply(classificar))
    st.subheader("Histórico de Rodadas")
    st.dataframe(df[::-1], use_container_width=True)

    # Gráfico
    st.subheader("Gráfico das Rodadas")
    fig, ax = plt.subplots()
    ax.plot(df["Valor"], marker='o', label="Valor")
    media_movel = df["Valor"].rolling(window=3).mean()
    ax.plot(media_movel, linestyle='--', color='orange', label="Média Móvel")
    ax.set_title("Evolução dos Valores")
    ax.set_ylabel("Multiplicador")
    ax.set_xlabel("Rodadas")
    ax.legend()
    st.pyplot(fig)

    # Estatísticas
    media_simples = round(df["Valor"].mean(), 2)
    pesos = np.arange(1, len(df)+1)
    media_ponderada = round(np.average(df["Valor"], weights=pesos), 2)

    st.subheader("Análise Inteligente")
    st.markdown(f"**Média Simples:** `{media_simples}`")
    st.markdown(f"**Média Ponderada:** `{media_ponderada}`")

    # IA - Previsão com regressão linear simples
    def prever_proxima_rodada(valores):
        if len(valores) < 3:
            return round(np.mean(valores), 2)
        x = np.arange(len(valores))
        y = np.array(valores)
        coef = np.polyfit(x, y, 1)
        return round(coef[0] * (len(valores)) + coef[1], 2)

    proxima = prever_proxima_rodada(df["Valor"].tolist())
    risco, emoji = classificar(proxima)

    st.markdown(f"**Previsão da Próxima Rodada:** `{proxima}` {emoji}")
    st.markdown(f"**Nível de Risco Previsto:** `{risco}`")

    # Exportar
    st.download_button("⬇️ Baixar CSV", data=df.to_csv(index=False).encode('utf-8'),
                       file_name="historico_aviator.csv", mime="text/csv")

    # Limpar histórico
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.dados = []
        st.success("Histórico apagado.")

else:
    st.info("Ainda não há rodadas registradas.")
