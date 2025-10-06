import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
import time
import matplotlib.pyplot as plt

# Importações de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

# Configuração da Página
st.set_page_config(
    page_title="Ponto do Bicho ML App",
    layout="wide"
)

# Título Principal do Aplicativo
st.title("Sistema Integrado de Scraping e Previsão de ML")

# ----------------------------------------------------------------------
# NOVO: Inicialização do Session State
# ----------------------------------------------------------------------
if 'df_raspado' not in st.session_state:
    st.session_state['df_raspado'] = pd.DataFrame() # DataFrame vazio inicialmente

# ----------------------------------------------------------------------
# 1. Funções de Web Scraping
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600) # Caches data for 1 hour
def pegar_resultados(state: str, date: str):
    """Busca resultados da API e retorna um DataFrame do Pandas."""
    url = "https://api.pontodobicho.com/bets/jb/results"
    params = {"state": state, "date": date}

    try:
        with st.spinner(f"Buscando dados de {state} em {date}..."):
            resp = requests.get(url, params=params, timeout=15)
            
            if resp.status_code != 200:
                st.error(f"Falha de API (Status {resp.status_code}) para {state} em {date}")
                return None
            
            data = resp.json()
            
            if "data" not in data or not data["data"]:
                st.info(f"Nenhum resultado encontrado para {state} em {date}.")
                return None

            registros = []
            for r in data["data"]:
                nome = r.get("lotteryName", "")
                horario_match = re.search(r'(\d{2}h)', nome)
                horario = horario_match.group(1) if horario_match else ""

                places = r.get("places", [])
                for idx, valor in enumerate(places, start=1):
                    try:
                        valor_numerico = int(valor)
                    except ValueError:
                        continue 

                    registros.append({
                        "lotteryName": nome,
                        "horario": horario,
                        "posicao": idx,
                        "valor": valor_numerico,
                        "state": state,
                        "date": date
                    })

            return pd.DataFrame(registros)

    except requests.exceptions.Timeout:
        st.error(f"Tempo limite excedido ao buscar dados de {state} em {date}.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
        return None

def gerar_ultimos_dias(n_dias=1):
    """Gera uma lista de datas para scraping."""
    hoje = datetime.today()
    datas = [(hoje - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_dias)]
    datas.reverse()
    return datas

def realizar_scraping_e_coletar_dados(estados, n_dias, intervalo_scraping):
    """Função principal que orquestra o scraping."""
    datas = gerar_ultimos_dias(n_dias)
    todos_dados = []
    total_de_consultas = len(estados) * len(datas)
    
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    consulta_count = 0
    
    for estado in estados:
        for data in datas:
            consulta_count += 1
            percent_complete = consulta_count / total_de_consultas
            status_text.text(f"Progresso do Scraping: {consulta_count}/{total_de_consultas} ({estado} - {data})")
            progress_bar.progress(percent_complete)

            df = pegar_resultados(estado, data)
            if df is not None:
                todos_dados.append(df)
            
            time.sleep(intervalo_scraping) 

    progress_container.empty()
    
    if todos_dados:
        df_total = pd.concat(todos_dados, ignore_index=True)
        st.success("Scraping concluído com sucesso!")
        return df_total
    else:
        st.warning("Nenhum dado coletado após o scraping.")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 2. Funções de Machine Learning
# ----------------------------------------------------------------------

@st.cache_data
def treinar_e_prever(df_total: pd.DataFrame, janela: int, n_passos: int, colunas_ml: list, modelo_selecionado: str):
    """Treina o modelo ML escolhido e faz previsões, aplicando limites ao resultado."""
    
    if len(df_total) < janela:
        return None, None, None, None
        
    # 1. Preparação dos Dados
    dados = df_total[colunas_ml].values
    
    # 2. Normalização
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(dados)

    # 3. Criação de Janelas (Função interna)
    def criar_janelas(series, janela=3):
        X, y = [], []
        for i in range(len(series) - janela):
            X.append(series[i:i+janela].flatten()) 
            y.append(series[i+janela])            
        return np.array(X), np.array(y)
        
    X, y = criar_janelas(dados_norm, janela)
    
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 4. Separar treino e teste
    test_size = 0.2
    if len(X) * test_size < 1:
        test_size = 1 / len(X)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # 5. Modelo de machine learning e Treinamento
    score = 0
    with st.spinner(f"Treinando o modelo {modelo_selecionado}..."):
        if modelo_selecionado == 'MLPRegressor':
            modelo = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42)
            modelo.fit(X_train, y_train)
        elif modelo_selecionado == 'SVR':
            if len(colunas_ml) > 1:
                st.warning("O **SVR** suporta apenas 1 coluna de previsão. Usando apenas a primeira coluna.")
                y_train_svr = y_train[:, 0]
                y_test_svr = y_test[:, 0]
            else:
                y_train_svr = y_train.flatten()
                y_test_svr = y_test.flatten()

            modelo = SVR(kernel='rbf', C=100)
            modelo.fit(X_train, y_train_svr)
            
            y_pred_test = modelo.predict(X_test)
            score = r2_score(y_test_svr, y_pred_test)
            
        elif modelo_selecionado == 'RandomForestRegressor':
            modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            modelo.fit(X_train, y_train)
        elif modelo_selecionado == 'LinearRegression':
            modelo = LinearRegression(n_jobs=-1)
            modelo.fit(X_train, y_train)
        
        if modelo_selecionado != 'SVR':
             score = modelo.score(X_test, y_test)
        
        st.info(f"Modelo: **{modelo_selecionado}** | Acurácia (**R² no teste**): **{score:.4f}**")

    # 6. Predição futura
    entrada = dados_norm[-janela:].flatten().reshape(1, -1)
    previsoes = []
    dados_hist_pred = dados_norm.copy()

    for _ in range(n_passos):
        pred = modelo.predict(entrada)[0]
        
        if modelo_selecionado == 'SVR':
            if len(colunas_ml) == 1:
                previsoes.append([pred])
                pred_reshaped = np.array([pred]).reshape(1, 1)
            else:
                pred_full = [0] * len(colunas_ml)
                pred_full[0] = pred
                previsoes.append(pred_full)
                pred_reshaped = np.array(pred_full).reshape(1, -1)
        else:
            previsoes.append(pred)
            pred_reshaped = pred.reshape(1, -1)

        dados_hist_pred = np.vstack([dados_hist_pred, pred_reshaped])
        entrada = dados_hist_pred[-janela:].flatten().reshape(1, -1)

    # Reverter normalização
    previsoes_inversa = scaler.inverse_transform(np.array(previsoes).reshape(-1, len(colunas_ml)))

    # Aplicando Limite e Arredondamento
    previsoes_finais = np.maximum(0, previsoes_inversa)
    previsoes_finais = np.minimum(9999, np.round(previsoes_finais)).astype(int)
    
    # Construir DataFrame de Previsões
    df_previsoes = pd.DataFrame(previsoes_finais, columns=colunas_ml)
    df_previsoes['Passo'] = range(1, n_passos + 1)
    
    df_hist = pd.DataFrame(dados, columns=colunas_ml)

    return df_previsoes, df_hist, modelo, score

# ----------------------------------------------------------------------
# 3. Interface do Streamlit (Sidebar)
# ----------------------------------------------------------------------

estados_default = ["PO", "RJ"]
st.sidebar.header("Parâmetros do Aplicativo")

# --- 1. Parâmetros de Scraping ---
st.sidebar.subheader("1. Scraping de Dados")
n_dias_scraping = st.sidebar.slider("Número de dias para buscar dados", 1, 90, 7)

intervalo_scraping = st.sidebar.slider(
    "Intervalo (segundos) entre as consultas de API", 
    0.1, 
    5.0, 
    0.5, 
    step=0.1,
    format="%.1f segundos"
)

estados_selecionados = st.sidebar.multiselect(
    "Estados para Scraping",
    options=["PO", "RJ", "SP", "MG", "ES", "PR", "SC"],
    default=estados_default
)

# NOVO BOTÃO PARA APENAS SCRAPING
if st.sidebar.button("1. Executar Scraping e Salvar Dados"):
    if not estados_selecionados:
        st.error("Por favor, selecione pelo menos um estado para o scraping.")
        st.stop()
        
    st.header("1. Resultados do Web Scraping")
    df_raspado = realizar_scraping_e_coletar_dados(
        estados_selecionados, n_dias_scraping, intervalo_scraping
    )
    
    if not df_raspado.empty:
        # ARMAZENA O RESULTADO NO SESSION STATE
        st.session_state['df_raspado'] = df_raspado
        st.subheader("Dados Raspados (Amostra)")
        st.dataframe(df_raspado.tail(10)) 


# --- 2. Parâmetros de Machine Learning ---
st.sidebar.subheader("2. Parâmetros de Machine Learning")

modelo_selecionado = st.sidebar.radio(
    "Escolha o Algoritmo de ML",
    ('MLPRegressor', 'SVR', 'RandomForestRegressor', 'LinearRegression')
)

colunas_ml = st.sidebar.multiselect(
    "Colunas para a Previsão (ML)",
    options=['valor', 'posicao'],
    default=['valor']
)
janela_ml = st.sidebar.slider("Tamanho da Janela (linhas históricas)", 10, 200, 100)
n_passos_previsao = st.sidebar.slider("Passos futuros para prever", 1, 10, 5)


# ----------------------------------------------------------------------
# 4. Bloco de Execução de Machine Learning (AUTOMÁTICO)
# ----------------------------------------------------------------------
st.header("2. Previsão de Machine Learning")

df_raspado = st.session_state['df_raspado']

if df_raspado.empty:
    st.info("Aguardando dados de scraping. Clique em '1. Executar Scraping e Salvar Dados' na barra lateral.")
else:
    # Validação e Execução de ML
    if not colunas_ml:
        st.error("Por favor, selecione pelo menos uma coluna para ML.")
    elif len(df_raspado) < janela_ml:
        st.warning(f"Dados raspados insuficientes ({len(df_raspado)} linhas) para o tamanho da Janela ML ({janela_ml}). Reduza a janela.")
    else:
        # Dados válidos, executa o ML automaticamente
        
        # Ordenar os dados
        df_ml = df_raspado.sort_values(by=['date', 'horario', 'posicao']).reset_index(drop=True)
        
        # Treinar e Prever
        df_previsoes, df_hist, modelo, score = treinar_e_prever(
            df_ml, janela_ml, n_passos_previsao, colunas_ml, modelo_selecionado
        )

        if df_previsoes is not None:
            # 3. Mostrar Previsões
            st.subheader(f"Previsões Futuras (Modelo: {modelo_selecionado})")
            st.dataframe(df_previsoes) 
            
            # 4. Plotar Resultados
            if colunas_ml:
                coluna_plot = colunas_ml[0]
                
                st.subheader(f"Gráfico de Histórico e Previsão para '{coluna_plot}'")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Plotar Histórico
                ax.plot(df_hist.index, df_hist[coluna_plot], label="Histórico")
                
                # Plotar Previsão
                indice_previsao = range(len(df_hist), len(df_hist) + n_passos_previsao)
                ax.plot(indice_previsao, df_previsoes[coluna_plot], "r--", label="Previsão")
                
                ax.set_xlabel("Índice (Amostra)")
                ax.set_ylabel(coluna_plot)
                ax.legend()
                ax.set_title(f"Modelo: {modelo_selecionado} | R²: {score:.4f}")
                
                st.pyplot(fig)
