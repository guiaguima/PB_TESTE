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
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from scipy.stats import mode # Usado para encontrar o valor mais frequente

# Configuração da Página
st.set_page_config(
    page_title="Ponto do Bicho ML App",
    layout="wide"
)

# Título Principal do Aplicativo
st.title("Sistema Integrado de Scraping e Previsão de ML")

# ----------------------------------------------------------------------
# Inicialização do Session State para armazenar dados
# ----------------------------------------------------------------------
if 'df_raspado' not in st.session_state:
    st.session_state['df_raspado'] = pd.DataFrame() 

# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE TRATAMENTO
# ----------------------------------------------------------------------

def milhar_para_bicho(valor_numerico: int) -> int:
    """Calcula o número do bicho (1 a 25) com base nos dois últimos dígitos do valor."""
    if valor_numerico < 0:
        return 0
    
    dezena = valor_numerico % 100
    
    if dezena == 0:
        return 25
    
    bicho = ((dezena - 1) // 4) + 1
    
    return min(bicho, 25) 

# ----------------------------------------------------------------------
# 1. Funções de Web Scraping
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600) # Caches data for 1 hour
def pegar_resultados(state: str, date: str):
    """Busca resultados da API e retorna um DataFrame do Pandas, convertendo a milhar para o número do Bicho."""
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
                        
                    numero_do_bicho = milhar_para_bicho(valor_numerico)

                    registros.append({
                        "lotteryName": nome,
                        "horario": horario,
                        "posicao": idx,
                        "valor": numero_do_bicho, 
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

def _criar_janelas(series_norm, series_original, janela=3, target_index=0):
    """Cria as janelas de features (X) normalizadas e o target (y_discrete) inteiro."""
    X, y_discrete = [], []
    for i in range(len(series_norm) - janela):
        X.append(series_norm[i:i+janela].flatten()) 
        y_discrete.append(series_original[i+janela, target_index])
    
    return np.array(X), np.array(y_discrete)

@st.cache_resource(max_entries=2) # Cacheia o modelo treinado para não retreinar
def treinar_modelo_e_prever(df_total: pd.DataFrame, janela: int, n_passos: int, colunas_ml: list, modelo_selecionado: str):
    """Treina um modelo de classificação e retorna as previsões futuras e o score."""
    
    if len(df_total) < janela:
        # Se os dados foram insuficientes, retorna None e None para ser tratado externamente
        return None, None, None
        
    dados = df_total[colunas_ml].values
    target_col_index = colunas_ml.index('valor')
    
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(dados)

    X, y_discrete = _criar_janelas(dados_norm, dados, janela, target_col_index)
    
    # Separar treino e teste
    test_size = 0.2
    if len(X) * test_size < 1:
        test_size = 1 / len(X)
        
    split_index = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_index], X[split_index:]
    
    # Zero-indexing o target (1-25) para (0-24)
    y_discrete_train = y_discrete[:split_index].astype(int) - 1
    y_discrete_test = y_discrete[split_index:].astype(int) - 1

    # 5. Modelo de machine learning e Treinamento
    if modelo_selecionado == 'RandomForestClassifier':
        modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif modelo_selecionado == 'XGBClassifier':
        modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Modelo não suportado: {modelo_selecionado}")
    
    modelo.fit(X_train, y_discrete_train) 
    
    # Avaliação (y_pred_test e y_discrete_test estão em 0-24)
    y_pred_test = modelo.predict(X_test)
    score = accuracy_score(y_discrete_test, y_pred_test)

    # 6. Predição futura (Multi-step prediction)
    entrada = dados_norm[-janela:].flatten().reshape(1, -1)
    previsoes_bicho = [] 
    dados_hist_pred = dados_norm.copy()

    for _ in range(n_passos):
        # Model predicts a value from 0 to 24
        pred_zero_indexed = modelo.predict(entrada)[0] 
        
        # Converte para Bicho (1-25)
        pred_bicho = pred_zero_indexed + 1
        previsoes_bicho.append(pred_bicho) 
        
        # Reshape logic para realimentar o próximo passo
        temp_pred_original_scale = dados[-janela:][-1:].copy() 
        temp_pred_original_scale[0, target_col_index] = pred_bicho 
        
        pred_reshaped = scaler.transform(temp_pred_original_scale)

        dados_hist_pred = np.vstack([dados_hist_pred, pred_reshaped])
        entrada = dados_hist_pred[-janela:].flatten().reshape(1, -1)
        
    # Retorna o array numpy das previsões de bicho (1-25) e o score.
    return np.array(previsoes_bicho).flatten(), score, pd.DataFrame(dados, columns=colunas_ml)


def compilar_previsoes(df_ml, janela_ml, n_passos_previsao, colunas_ml):
    """
    Executa o Random Forest e o XGBoost, compila os resultados e calcula a Moda
    (voto majoritário).
    """
    st.subheader("Processando e Consolidando Modelos...")
    
    # --- 1. Rodar RandomForest ---
    with st.spinner("Treinando e prevendo com RandomForestClassifier..."):
        rf_previsoes, rf_score, df_hist = treinar_modelo_e_prever(
            df_ml, janela_ml, n_passos_previsao, colunas_ml, 'RandomForestClassifier'
        )
    if rf_previsoes is None:
        return None, None, None, None
        
    st.info(f"RandomForestClassifier Acurácia: **{rf_score:.4f}**")
    
    # --- 2. Rodar XGBoost ---
    with st.spinner("Treinando e prevendo com XGBClassifier..."):
        xgb_previsoes, xgb_score, _ = treinar_modelo_e_prever(
            df_ml, janela_ml, n_passos_previsao, colunas_ml, 'XGBClassifier'
        )
    if xgb_previsoes is None:
        return None, None, None, None
        
    st.info(f"XGBClassifier Acurácia: **{xgb_score:.4f}**")

    # --- 3. Compilar Resultados ---
    
    # Cria o DataFrame de compilação
    df_compilado = pd.DataFrame({
        'Passo': range(1, n_passos_previsao + 1),
        'RF_Previsao': rf_previsoes.astype(int),
        'XGB_Previsao': xgb_previsoes.astype(int)
    })
    
    # Cria a função local para encontrar a moda (valor mais frequente)
    def encontrar_moda(row):
        """Encontra o valor mais frequente (moda) na linha, retornando o valor."""
        m = mode(row.values)
        # CORREÇÃO: Usa .flat[0] para extrair o valor de forma segura, 
        # seja m.mode um escalar ou um array de um elemento.
        return m.mode.flat[0] 

    # Aplica a função de moda linha a linha
    df_compilado['PREVISAO_CONSOLIDADA'] = df_compilado[['RF_Previsao', 'XGB_Previsao']].apply(encontrar_moda, axis=1).astype(int)
    
    return df_compilado, df_hist, rf_score, xgb_score


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

if st.sidebar.button("1. Executar Scraping e Salvar Dados"):
    if not estados_selecionados:
        st.error("Por favor, selecione pelo menos um estado para o scraping.")
        st.stop()
        
    st.header("1. Resultados do Web Scraping")
    df_raspado = realizar_scraping_e_coletar_dados(
        estados_selecionados, n_dias_scraping, intervalo_scraping
    )
    
    if not df_raspado.empty:
        st.session_state['df_raspado'] = df_raspado
        st.subheader(f"Dados Raspados (Amostra - {len(df_raspado)} linhas)")
        st.dataframe(df_raspado.tail(10)) 


# --- 2. Parâmetros de Machine Learning ---
st.sidebar.subheader("2. Parâmetros de Machine Learning")

st.sidebar.info("Os modelos **XGBoost** e **RandomForest** serão rodados e consolidados.")

# ----------------------------------------------------------------------
# 4. Bloco de Execução de Machine Learning (AUTOMÁTICO)
# ----------------------------------------------------------------------
st.header("2. Previsão de Machine Learning Consolidada")

df_raspado = st.session_state['df_raspado']

if df_raspado.empty:
    st.info("Aguardando dados de scraping. Clique em '1. Executar Scraping e Salvar Dados' na barra lateral.")
else:
    # ----------------------------------------------------------------
    # ENGENHARIA DE FEATURES (Criando Dia da Semana e Hora)
    # ----------------------------------------------------------------
    df_ml = df_raspado.copy()
    
    df_ml['date_obj'] = pd.to_datetime(df_ml['date'])
    df_ml['dia_semana'] = df_ml['date_obj'].dt.dayofweek
    
    df_ml['hora_sorteio_limpo'] = df_ml['horario'].str.replace('h', '')
    
    df_ml = df_ml[df_ml['hora_sorteio_limpo'] != ''].copy()
    
    df_ml['hora_sorteio'] = df_ml['hora_sorteio_limpo'].astype(int)
    
    df_ml.drop(columns=['hora_sorteio_limpo', 'date_obj'], errors='ignore', inplace=True)
    
    df_ml = df_ml.sort_values(by=['date', 'horario', 'posicao']).reset_index(drop=True)
    
    # ----------------------------------------------------------------
    
    colunas_ml = st.sidebar.multiselect(
        "Colunas para a Previsão (ML)",
        options=['valor', 'posicao', 'dia_semana', 'hora_sorteio'],
        default=['valor'] 
    )
    
    if 'valor' not in colunas_ml:
         st.error("A coluna **'valor'** (o bicho) é obrigatória para o treinamento do modelo.")
         st.stop()
         
    janela_ml = st.sidebar.slider("Tamanho da Janela (linhas históricas)", 10, 200, 100, key='janela')
    n_passos_previsao = st.sidebar.slider("Passos futuros para prever", 1, 10, 5, key='passos')

    
    # Validação e Execução de ML
    if not colunas_ml:
        st.error("Por favor, selecione pelo menos uma coluna para ML.")
    elif len(df_ml) < janela_ml:
        st.warning(f"Dados insuficientes ({len(df_ml)} linhas) para a Janela ML ({janela_ml}). Reduza a janela.")
    else:
        # Treinar, Prever e Compilar os dois modelos
        df_compilado, df_hist, rf_score, xgb_score = compilar_previsoes(
            df_ml, janela_ml, n_passos_previsao, colunas_ml
        )

        if df_compilado is not None:
            # 3. Mostrar Tabela Consolidada
            st.subheader("Tabela de Voto Majoritário e Previsões")
            st.markdown("A **PREVISAO_CONSOLIDADA** mostra o valor mais frequente entre RF e XGB.")
            
            # Formatação para destacar a previsão consolidada
            def highlight_consolidated(s):
                # O índice 3 é a coluna 'PREVISAO_CONSOLIDADA'
                is_consolidated = s.index == 3 
                return ['background-color: yellow' if v else '' for v in is_consolidated]

            st.dataframe(df_compilado.style.apply(highlight_consolidated, axis=1, subset=df_compilado.columns))
            
            # 4. Plotar Resultados (Plotamos apenas a previsão consolidada)
            coluna_plot = 'valor'
            
            st.subheader(f"Gráfico de Histórico e Previsão Consolidada para '{coluna_plot}' (Bicho)")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plota o Histórico (o bicho real)
            ax.plot(df_hist.index, df_hist[coluna_plot], label="Histórico")
            
            # Plota a Previsão Consolidada
            indice_previsao = range(len(df_hist), len(df_hist) + n_passos_previsao)
            ax.plot(indice_previsao, df_compilado['PREVISAO_CONSOLIDADA'], "g--", label="Previsão Consolidada (Moda)")
            
            ax.set_xlabel("Índice (Amostra)")
            ax.set_ylabel(coluna_plot)
            ax.legend()
            ax.set_title(f"Previsão Consolidada (RF Score: {rf_score:.4f}, XGB Score: {xgb_score:.4f})")
            
            st.pyplot(fig)
