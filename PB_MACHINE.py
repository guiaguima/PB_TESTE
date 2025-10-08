import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
import time 
import matplotlib.pyplot as plt
from collections import defaultdict 
import os 
import random 

# --- NOVO IMPORT DE DEEP LEARNING ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM # Adicionado LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping # Adicionado EarlyStopping
# ------------------------------------

# Importações de Machine Learning Clássico
from sklearn.model_selection import TimeSeriesSplit
# ALTERADO PARA STANDARDS SCALER
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight # Mantido para ponderação

# --- CONSTANTES DE CONFIGURAÇÃO ---
N_ITERATIONS = 50 
N_CLASSES = 25 
JANELA_ML_DEFAULT = 100 
MARGEM_SEGURANCA_JANELA = 20
N_PASSOS_PREVISAO = 1 
FILTRO_POSICAO_MAX = 5 

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ponto do Bicho ML Automatizado (LSTM Otimizada)",
    layout="wide"
)
st.title("Sistema Automatizado de LSTM Otimizada com Ensemble Voting")
st.caption("Implementadas: LSTM, StandardScaler e Early Stopping.")


# ----------------------------------------------------------------------
# INICIALIZAÇÃO DO SESSION STATE
# ----------------------------------------------------------------------
if 'df_raspado' not in st.session_state:
    st.session_state['df_raspado'] = pd.DataFrame() 
if 'ensemble_results' not in st.session_state:
    st.session_state['ensemble_results'] = None

# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE TRATAMENTO
# ----------------------------------------------------------------------

def milhar_para_bicho(valor_numerico: int) -> int:
    """Calcula o número do bicho (1 a 25)."""
    if valor_numerico < 0: return 0
    dezena = valor_numerico % 100
    if dezena == 0: return 25
    bicho = ((dezena - 1) // 4) + 1
    return min(bicho, 25) 

def bicho_para_grupo(bicho: int) -> int:
    """Mapeia o bicho (1-25) para o número do grupo (1-25)."""
    if bicho < 1 or bicho > 25: return 0
    return bicho

def gerar_parametros_aleatorios(n_iteracoes):
    """Gera n_iteracoes conjuntos de hiperparâmetros de entrada e Keras aleatórios."""
    parametros = []
    for _ in range(n_iteracoes):
        params = {
            # Hiperparâmetros de Feature Engineering
            'n_lag_features': random.randint(3, 10),
            'posicao_selecionada': random.choice(['Todos'] + list(range(1, FILTRO_POSICAO_MAX + 1))),
            'janela_minima': random.randint(50, 200),
            
            # Hiperparâmetros do Keras
            'n_lstm_layers': random.randint(1, 2), # 1 ou 2 Camadas LSTM
            'n_units': random.choice([64, 128, 256]),
            'learning_rate': random.choice([0.01, 0.005, 0.001]),
            'epochs': 50, # Mantido fixo, Early Stopping irá parar antes
            'batch_size': random.choice([16, 32, 64]),
        }
        parametros.append(params)
    return parametros

# ----------------------------------------------------------------------
# 1. Funções de Web Scraping (Mantidas)
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600)
def pegar_resultados(state: str, date: str):
    # Implementação de scraping mantida...
    url = "https://api.pontodobicho.com/bets/jb/results"
    params = {"state": state, "date": date}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200: return None 
        data = resp.json()
        if "data" not in data or not data["data"]: return None
        registros = []
        for r in data["data"]:
            nome = r.get("lotteryName", "")
            horario_match = re.search(r'(\d{2}h)', nome)
            horario = horario_match.group(1) if horario_match else "" 
            places = r.get("places", [])
            for idx, valor in enumerate(places, start=1):
                if idx > FILTRO_POSICAO_MAX: continue
                try: valor_numerico = int(valor)
                except ValueError: continue 
                numero_do_bicho = milhar_para_bicho(valor_numerico)
                registros.append({"lotteryName": nome, "horario": horario, "posicao": idx, "valor": numero_do_bicho, "state": state, "date": date})
        return pd.DataFrame(registros)
    except Exception: return None

def gerar_ultimos_dias(n_dias=1):
    hoje = datetime.today()
    datas = [(hoje - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_dias)]
    datas.reverse()
    return datas

def realizar_scraping_e_coletar_dados(estados, n_dias):
    """Orquestra o scraping."""
    datas = gerar_ultimos_dias(n_dias)
    tarefas = [(estado, data) for estado in estados for data in datas]
    todos_dados = []
    
    status_msg = st.empty()
    
    for estado, data in tarefas:
        status_msg.text(f"Scraping: Processando {estado} - {data}...")
        try:
            df = pegar_resultados(estado, data)
            if df is not None: todos_dados.append(df)
        except Exception as exc:
            status_msg.error(f'Erro de Scraping em {estado} - {data}: {exc}')
        time.sleep(0.1) 

    status_msg.empty()
    
    if todos_dados:
        df_total = pd.concat(todos_dados, ignore_index=True)
        st.success(f"Scraping concluído! {len(df_total)} registros coletados.")
        return df_total
    else:
        st.warning("Nenhum dado coletado após o scraping.")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# 2. Funções de Pré-Processamento e Preparação de ML (Ajustadas para LSTM)
# ----------------------------------------------------------------------

def _pre_processar_dados_ml(df_raspado, posicao_selecionada, n_lag_features, janela_minima):
    """Executa engenharia de features e retorna o df processado e a janela adaptativa."""
    df_ml = df_raspado.copy()
    
    # Processamento e Filtro de Contexto
    df_ml['date_obj'] = pd.to_datetime(df_ml['date'])
    df_ml['dia_semana'] = df_ml['date_obj'].dt.dayofweek
    df_ml['hora_sorteio_limpo'] = df_ml['horario'].str.replace('h', '')
    df_ml = df_ml[df_ml['hora_sorteio_limpo'] != ''].copy()
    df_ml['hora_sorteio'] = df_ml['hora_sorteio_limpo'].astype(int)
    
    if posicao_selecionada != 'Todos':
        df_ml = df_ml[df_ml['posicao'] == posicao_selecionada].copy()
        
    df_ml = df_ml.sort_values(by=['date_obj', 'hora_sorteio', 'posicao'], ignore_index=True)
    df_ml.reset_index(drop=True, inplace=True) 

    # Geração de Features (Lag)
    cols_lag, cols_grupo_lag = [], []
    for i in range(1, n_lag_features + 1): 
        lag_col, grupo_lag_col = f'valor_lag_{i}', f'grupo_lag_{i}'
        df_ml[lag_col] = df_ml['valor'].shift(i)
        df_ml[grupo_lag_col] = df_ml[lag_col].apply(bicho_para_grupo)
        cols_lag.append(lag_col); cols_grupo_lag.append(grupo_lag_col)
    
    # Recência 
    recencia_data = defaultdict(lambda: 0)
    for bicho in range(1, 26): df_ml[f'recencia_bicho_{bicho}'] = 0
    for index, row in df_ml.iterrows():
        bicho_atual = row['valor']
        for bicho in range(1, 26): recencia_data[bicho] += 1
        recencia_data[bicho_atual] = 0
        for bicho in range(1, 26): df_ml.loc[index, f'recencia_bicho_{bicho}'] = recencia_data[bicho]


    # Remove linhas iniciais onde features não estão completas
    df_ml = df_ml.iloc[max(25, n_lag_features + 1):].copy() 
    df_ml.reset_index(drop=True, inplace=True) 

    # OHE e Limpeza Final
    cols_to_ohe = cols_lag + cols_grupo_lag
    df_ml.dropna(subset=cols_to_ohe, inplace=True) 
    
    for col in cols_to_ohe: df_ml[col] = df_ml[col].astype(int).astype(str) 
    df_ml = pd.get_dummies(df_ml, columns=cols_to_ohe, prefix=cols_to_ohe, drop_first=True) 
    
    cols_to_drop = ['hora_sorteio_limpo', 'date_obj', 'lotteryName', 'horario', 'date']
    df_ml.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    cols_ohe_context = ['posicao', 'dia_semana', 'hora_sorteio', 'state']
    if posicao_selecionada != 'Todos':
        cols_ohe_context.remove('posicao'); df_ml.drop(columns=['posicao'], errors='ignore', inplace=True)
    df_ml = pd.get_dummies(df_ml, columns=cols_ohe_context, prefix=cols_ohe_context, drop_first=True) 
    
    df_ml.dropna(inplace=True); df_ml.reset_index(drop=True, inplace=True)
    
    colunas_ml = [col for col in df_ml.columns if col not in ['valor']]
    colunas_ml.insert(0, 'valor') 

    # CÁLCULO DA JANELA ADAPTATIVA
    max_janela_possivel = len(df_ml) - MARGEM_SEGURANCA_JANELA
    
    if max_janela_possivel <= 0:
        janela_adaptativa = 0
    elif max_janela_possivel < janela_minima:
        janela_adaptativa = max_janela_possivel
    else:
        janela_adaptativa = janela_minima

    return df_ml, colunas_ml, janela_adaptativa

def _criar_janelas(series_norm, series_original, janela, target_col_index=0):
    """Cria janelas no formato 3D (amostras, janelas, features) para LSTM."""
    X, y_discrete = [], []
    n_features = series_norm.shape[1]
    
    for i in range(len(series_norm) - janela):
        # A nova entrada é um bloco 2D (janela, n_features)
        X.append(series_norm[i:i+janela].reshape(janela, n_features)) 
        y_discrete.append(series_original[i+janela, target_col_index])
    
    # X precisa ser (amostras, janela, features)
    return np.array(X), np.array(y_discrete)


def _preparar_dados_ml(df_ml, colunas_ml, janela_ml):
    """Prepara X, y (OHE) e calcula sample_weight. Usa StandardScaler."""
    dados = df_ml[colunas_ml].values
    target_col_index = colunas_ml.index('valor')
    
    # MELHORIA 2: StandardScaler para padronização
    scaler = StandardScaler()
    dados_norm = scaler.fit_transform(dados)

    X, y_discrete = _criar_janelas(dados_norm, dados, janela_ml, target_col_index=target_col_index)
    
    if len(X) < 2: return None 
    
    # Divisão TimeSeries
    tscv = TimeSeriesSplit(n_splits=2) 
    # Pegamos o último split (o mais recente)
    train_index, test_index = list(tscv.split(X))[-1] 
    
    X_train, X_test = X[train_index], X[test_index]
    y_train_int = y_discrete[train_index].astype(int) - 1 
    y_test_int = y_discrete[test_index].astype(int) - 1   

    # Conversão para One-Hot Encoding para o Keras
    y_train = to_categorical(y_train_int, num_classes=N_CLASSES)
    y_test = to_categorical(y_test_int, num_classes=N_CLASSES)

    # Cálculo de PONDERAÇÃO ADAPTATIVA (Temporal + Classe)
    train_size = len(X_train)
    weights_exp = np.exp(np.linspace(0, 2, train_size)) 
    weights_exp = weights_exp / np.sum(weights_exp) * train_size
    
    weights_class = compute_sample_weight(class_weight='balanced', y=y_train_int)
    sample_weight_train = weights_exp * weights_class
    
    return X_train, y_train, X_test, y_test, dados_norm, dados, target_col_index, scaler, sample_weight_train, y_test_int


def criar_modelo_keras(input_shape, params):
    """Constrói o modelo Keras com camadas LSTM (MELHORIA 1)."""
    tf.keras.backend.clear_session()
    model = Sequential()
    
    # Camada(s) LSTM
    for i in range(params['n_lstm_layers']):
        return_sequences = (i < params['n_lstm_layers'] - 1) # True para todas, exceto a última
        model.add(LSTM(
            units=params['n_units'], 
            return_sequences=return_sequences, 
            input_shape=input_shape if i == 0 else None
        ))
        model.add(Dropout(0.2))

    # Camada de Saída
    model.add(Dense(N_CLASSES, activation='softmax'))
    
    # Compilação
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


def _prever_passo_1_keras(modelo, dados_norm, dados, target_col_index, scaler, janela_ml):
    """Gera a previsão para APENAS o Passo 1 usando o modelo Keras (LSTM)."""
    n_features = dados_norm.shape[1]
    
    # A entrada final é o bloco 3D (1, janela, features)
    entrada = dados_norm[-janela_ml:].reshape(1, janela_ml, n_features)
    
    # Previsão
    proba = modelo.predict(entrada, verbose=0)[0]
    pred_zero_indexed = np.argmax(proba) 
    pred_bicho = pred_zero_indexed + 1
    
    return pred_bicho, proba

# ----------------------------------------------------------------------
# 3. FUNÇÃO PRINCIPAL DE ENSEMBLE E VOTING
# ----------------------------------------------------------------------

def executar_ensemble_voting(df_raspado):
    
    parametros_a_testar = gerar_parametros_aleatorios(N_ITERATIONS)
    
    all_predictions = []
    all_accuracies = []
    
    status_bar = st.progress(0, text=f"Iniciando {N_ITERATIONS} Testes de Keras/LSTM...")
    
    # MELHORIA 3: Early Stopping (para evitar overfitting e economizar tempo)
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=5,          # Para se a acurácia não melhorar por 5 épocas
        restore_best_weights=True,
        mode='max'
    )
    
    for i, params in enumerate(parametros_a_testar):
        
        status_bar.progress((i + 1) / N_ITERATIONS, text=f"Executando Teste {i + 1}/{N_ITERATIONS}. Lags: {params['n_lag_features']}, Unidades: {params['n_units']}...")
        
        # 1. PRÉ-PROCESSAMENTO 
        df_ml, colunas_ml, JANELA_ML_ADAPTATIVA = _pre_processar_dados_ml(
            df_raspado, 
            posicao_selecionada=params['posicao_selecionada'], 
            n_lag_features=params['n_lag_features'],
            janela_minima=params['janela_minima']
        )
        
        if JANELA_ML_ADAPTATIVA <= 0: continue
            
        # 2. PREPARAÇÃO (Criação de Janelas 3D e OHE)
        data_prep_result = _preparar_dados_ml(df_ml, colunas_ml, JANELA_ML_ADAPTATIVA)
        
        if data_prep_result is None: continue
            
        X_train, y_train, X_test, y_test, dados_norm, dados, target_col_index, scaler, sample_weight_train_full, y_test_int = data_prep_result
        
        # Input Shape para LSTM: (Janela, Features)
        input_shape = (X_train.shape[1], X_train.shape[2]) 
        
        # 3. TREINAMENTO (Keras LSTM)
        modelo = criar_modelo_keras(input_shape, params)
        
        try:
            modelo.fit(
                X_train, y_train, 
                epochs=params['epochs'], 
                batch_size=params['batch_size'], 
                sample_weight=sample_weight_train_full, 
                validation_data=(X_test, y_test), # Usando Teste como Validação
                callbacks=[early_stopping],        # Adicionando Early Stopping
                verbose=0 
            )
        except Exception:
            st.warning(f"Teste {i+1} ignorado: Erro no treinamento do Keras/LSTM.")
            continue


        # 4. AVALIAÇÃO E PREVISÃO (Passo 1)
        y_pred_test_ohe = modelo.predict(X_test, verbose=0)
        y_pred_test_int = np.argmax(y_pred_test_ohe, axis=1)

        test_score = accuracy_score(y_test_int, y_pred_test_int)
        
        pred_bicho, proba = _prever_passo_1_keras(
            modelo, dados_norm, dados, target_col_index, scaler, JANELA_ML_ADAPTATIVA
        )
        
        all_predictions.append(pred_bicho)
        all_accuracies.append(test_score)
        
        # Limpar memória do Keras/TF (Crucial em loops)
        del modelo
        tf.keras.backend.clear_session()


    status_bar.empty()
    st.success(f"Ensemble Voting concluído após {len(all_predictions)} testes válidos.")
    
    if not all_predictions:
        return None

    # VOTAÇÃO MAJORITÁRIA
    votes_series = pd.Series(all_predictions)
    bicho_consenso = votes_series.mode()[0] 
    
    contagem_votos = votes_series.value_counts().sort_values(ascending=False)
    votos_vencedor = contagem_votos.iloc[0]
    confianca_votos = (votos_vencedor / len(all_predictions)) * 100
    
    top5_votos = contagem_votos.head(5).reset_index()
    top5_votos.columns = ['Bicho', 'Total de Votos']
    top5_votos['Percentual (%)'] = (top5_votos['Total de Votos'] / len(all_predictions) * 100).round(2)
    
    return {
        'bicho_consenso': bicho_consenso,
        'confianca_votos': confianca_votos,
        'media_acuracia': np.mean(all_accuracies),
        'total_votos': len(all_predictions),
        'top5_votos': top5_votos,
        'all_predictions': votes_series.to_frame(name='Bicho Previsto')
    }

# ----------------------------------------------------------------------
# 4. EXECUÇÃO DO STREAMLIT 
# ----------------------------------------------------------------------

# --- PARÂMETROS FIXOS PARA EXECUÇÃO (AJUSTADOS) ---
DIAS_SCRAPING = 21
ESTADOS_SCRAPING = ["RJ"]

st.markdown("---")
st.subheader("1. Coleta e Preparação Inicial de Dados")
st.info(f"Parâmetros fixos: **{DIAS_SCRAPING} dias** de dados e **apenas o estado {ESTADOS_SCRAPING[0]}**.")


# Execução do Scraping
if st.button("1. Iniciar Scraping de Dados"):
    st.session_state['df_raspado'] = realizar_scraping_e_coletar_dados(ESTADOS_SCRAPING, DIAS_SCRAPING)
    st.session_state['ensemble_results'] = None 
    if not st.session_state['df_raspado'].empty:
        st.info(f"Dados brutos coletados: {len(st.session_state['df_raspado'])} registros. Pronto para Ensemble Voting.")
        st.dataframe(st.session_state['df_raspado'].tail(10))

# Execução do Ensemble Voting
if not st.session_state['df_raspado'].empty:
    st.markdown("---")
    st.subheader(f"2. Ensemble Voting Automatizado com LSTM ({N_ITERATIONS} Modelos)")
    st.info(f"O sistema irá treinar {N_ITERATIONS} modelos de **LSTM** com Padronização (StandardScaler) e Early Stopping.")

    if st.button(f"2. Executar {N_ITERATIONS} Testes e Votação"):
        st.session_state['ensemble_results'] = executar_ensemble_voting(st.session_state['df_raspado'])
        st.rerun()

# --- EXIBIÇÃO DE RESULTADOS FINAIS ---
if st.session_state['ensemble_results']:
    results = st.session_state['ensemble_results']
    st.markdown("---")
    st.header("🏆 Resultado Final por Voto Majoritário (Consenso LSTM)")

    col_metricas, col_votos = st.columns([1, 2])

    with col_metricas:
        st.metric(
            label="Bicho de Consenso (Passo 1)", 
            value=results['bicho_consenso']
        )
        st.metric(
            label="Confiança do Voto (Total de Votos)", 
            value=f"{results['confianca_votos']:.2f}%", 
            delta=f"Total de {results['total_votos']} testes válidos"
        )
        st.metric(
            label="Média de Acurácia Teste (Ensemble)", 
            value=f"{results['media_acuracia']:.4f}"
        )
    
    with col_votos:
        st.subheader("Top 5 Bichos mais votados")
        st.dataframe(results['top5_votos'].style.background_gradient(cmap='YlGnBu', subset=['Percentual (%)']), hide_index=True)
    
    st.markdown("---")
    st.subheader("Distribuição Detalhada das 50 Previsões")
    
    # Plota a distribuição dos votos
    fig, ax = plt.subplots(figsize=(10, 5))
    results['all_predictions']['Bicho Previsto'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Frequência de Previsões por Bicho (50 Modelos LSTM)")
    ax.set_xlabel("Bicho Previsto"); ax.set_ylabel("Contagem de Votos")
    plt.xticks(rotation=0)
    st.pyplot(fig)


# --- Limpeza de Cache ---
st.markdown("---")
if st.button("Limpar Cache e Recarregar Programa"):
    st.cache_data.clear(); st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()
