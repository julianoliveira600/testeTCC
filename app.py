# app.py

import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# --- FUNÇÕES DO APLICATIVO ---

@st.cache_resource
def carrega_modelo():
    """
    Baixa nosso modelo treinado do Google Drive e o carrega na memória.
    O decorator @st.cache_resource garante que isso só aconteça uma vez.
    """
    # URL de compartilhamento do SEU modelo. Certifique-se que "Anyone with the link can view".
    # Link: /content/drive/MyDrive/TCC/testeTCC-modelos-treinados/modelo_quantizado16bits.tflite
    #https://drive.google.com/file/d/1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d/view?usp=drive_link

    # https://drive.google.com/file/d/1Gp3W6VwF4CpwQEmbAOuVW4mopXluVex9/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d'
    
    # Baixa o arquivo do Google Drive se ele ainda não existir
    gdown.download(url, 'modelo_quantizado16bits.tflite', quiet=False)
    
    # Carrega o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    
    return interpreter

def carrega_e_prepara_imagem():
    """
    Cria a interface de upload e pré-processa a imagem para o formato que nosso modelo espera.
    """
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Lê e abre a imagem
        image_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_data))

        st.image(pil_image, caption="Imagem Carregada", use_column_width=True)
        st.success('Imagem carregada com sucesso!')

        # --- Pré-processamento CRÍTICO ---
        # 1. Redimensiona a imagem para 224x224 pixels
        pil_image_resized = pil_image.resize((224, 224))
        
        # 2. Converte para um array numpy e normaliza (opcional, mas boa prática)
        image_array = np.array(pil_image_resized, dtype=np.float32)
        
        # 3. Adiciona uma dimensão de "lote" (batch) para o modelo
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    return None

def faz_previsao(interpreter, image_array):
    """
    Recebe o modelo e a imagem preparada, executa a predição e mostra os resultados.
    """
    # Pega os detalhes dos tensores de entrada e saída do modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define o tensor de entrada com a nossa imagem
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Executa a inferência
    interpreter.invoke()
    
    # Pega o resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # --- Lógica de Interpretação CORRIGIDA ---
    # Nosso modelo retorna um único score. score > 0.5 = maligno
    score = output_data[0][0]
    
    # Determina o diagnóstico principal
    if score < 0.5:
        st.write("## Diagnóstico: **Benigno**")
    else:
        st.write("## Diagnóstico: **Maligno**")

    # Calcula as probabilidades para o gráfico
    prob_benigno = 100 * (1 - score)
    prob_maligno = 100 * score
    
    # Cria um DataFrame para o gráfico
    df = pd.DataFrame({
        'Classe': ['Benigno', 'Maligno'],
        'Probabilidade (%)': [prob_benigno, prob_maligno]
    })
    
    # Cria e exibe o gráfico de barras com Plotly
    fig = px.bar(df,
                 y='Classe',
                 x='Probabilidade (%)',
                 orientation='h',
                 text=df['Probabilidade (%)'].apply(lambda x: f'{x:.2f}%'),
                 title='Confiança do Modelo no Diagnóstico',
                 range_x=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)

# --- FUNÇÃO PRINCIPAL ---
def main():
    """
    Função principal que organiza e executa o aplicativo.
    """
    # Configuração da página
    st.set_page_config(
        page_title="Sistema de Diagnóstico de Câncer",
        page_icon="🔬",
        layout="centered"
    )
    
    st.title("Sistema de Diagnóstico de Câncer 🔬")
    st.write("Faça o upload de uma imagem histopatológica para que o modelo de IA a classifique como benigna ou maligna.")

    # Carrega o modelo
    with st.spinner('Carregando modelo, por favor aguarde...'):
        interpreter = carrega_modelo()

    # Carrega a imagem
    image_array = carrega_e_prepara_imagem()
    
    # Se uma imagem foi carregada, faz a previsão
    if image_array is not None:
        faz_previsao(interpreter, image_array)

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()