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
    Baixa o modelo do Google Drive e o carrega na memória.
    """
    # Certifique-se que o link de compartilhamento está correto e com a permissão "Qualquer pessoa com o link".
    url = 'https://drive.google.com/uc?id=1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d'
    
    # CORREÇÃO: O nome da função é "download", não "dowload".
    output_filename = 'modelo_final_compativel.tflite'
    gdown.download(url, output_filename, quiet=False)
    
    interpreter = tf.lite.Interpreter(model_path=output_filename)
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    """
    Cria a interface de upload e pré-processa a imagem para o formato do modelo.
    """
    uploaded_file = st.file_uploader('Arraste e solte uma imagem ou clique para selecionar', type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None: 
        image_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_data))

        st.image(pil_image, caption="Imagem Carregada")
        st.success('Imagem carregada com sucesso!')

        # --- CORREÇÃO DE LÓGICA E SINTAXE ---
        # 1. Redimensionar a imagem PIL para 224x224 ANTES de converter para array.
        image_resized = pil_image.resize((224, 224))
        
        # 2. Converter a imagem JÁ REDIMENSIONADA para um array numpy.
        image_array = np.array(image_resized, dtype=np.float32)
        
        # 3. Adicionar a dimensão do "lote" (batch) no eixo 0. A variável "axis" foi substituída por 0.
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    return None

def previsao(interpreter, image_array):
    """
    Executa a predição e mostra os resultados em um gráfico.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # --- CORREÇÃO DA INTERPRETAÇÃO DO RESULTADO ---
    # O modelo retorna um único score. score > 0.5 significa maligno.
    score_maligno = output_data[0][0]
    score_benigno = 1 - score_maligno
    
    classes = ['Benigno', 'Maligno'] # CORREÇÃO: "malign" para "Maligno" para consistência.
    probabilidades = [score_benigno, score_maligno]

    df = pd.DataFrame({
        'classes': classes,
        'probabilidades (%)': [p * 100 for p in probabilidades]
    })
    
    fig = px.bar(df, 
                 y='classes', 
                 x='probabilidades (%)', 
                 orientation='h', 
                 text=df['probabilidades (%)'].apply(lambda x: f'{x:.2f}%'), 
                 title='Confiança do Modelo no Diagnóstico',
                 range_x=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """
    Função principal que organiza e executa o aplicativo.
    """
    st.set_page_config(
        page_title="Diagnóstico de Câncer Mamário em Animais",
        page_icon="🐾", # CORREÇÃO: Ícone de emoji
    )
    
    st.title("IA para Diagnóstico de Câncer Mamário em Animais Domésticos 🔬")

    # Carrega o modelo (com a chamada da função corrigida)
    # CORREÇÃO: Deve ser carrega_modelo() para EXECUTAR a função.
    with st.spinner('Carregando modelo de IA...'):
        interpreter = carrega_modelo()
    
    # Carrega a imagem
    image = carrega_imagem()
    
    # Faz a classificação se uma imagem foi carregada
    if image is not None:
        previsao(interpreter, image)

# Ponto de entrada do script
if __name__ == "__main__":
    main()