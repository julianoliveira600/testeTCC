# app.py - CÓDIGO DO PROJETO DE CÂNCER (ADAPTADO DO SEU EXEMPLO)

import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    """
    Baixa nosso modelo treinado do Google Drive e o carrega na memória.
    """
    # ==============================================================================
    # MUDANÇA 1: Usamos o link do nosso modelo final e compatível.
    #https://drive.google.com/file/d/1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d/view?usp=drive_link
    url ='https://drive.google.com/uc?id=1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d'
    # ==============================================================================
    
    output_filename = 'modelo_final_compativel.tflite'
    gdown.download(url, output_filename, quiet=False)
    
    interpreter = tf.lite.Interpreter(model_path=output_filename)
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    """
    Cria a interface de upload e pré-processa a imagem.
    """
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_data))

        st.image(pil_image, caption="Imagem Carregada")
        st.success('Imagem carregada com sucesso!')

        # ==============================================================================
        # MUDANÇA 2: Adicionamos o redimensionamento para 224x224.
        image_resized = pil_image.resize((224, 224))
        # ==============================================================================
        
        image_array = np.array(image_resized, dtype=np.float32)
        # A normalização /255.0 não é estritamente necessária para nosso modelo, mas não atrapalha.
        # image_array = image_array / 255.0
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

    # ==============================================================================
    # MUDANÇA 3: A lógica de interpretação do resultado é diferente.
    # O modelo de videira retornava 4 probabilidades. O nosso retorna 1 score.
    score_maligno = output_data[0][0]
    score_benigno = 1 - score_maligno
    
    classes = ['Benigno', 'Maligno']
    probabilidades = [score_benigno * 100, score_maligno * 100]
    # ==============================================================================
    
    df = pd.DataFrame({
        'classes': classes,
        'probabilidades (%)': probabilidades
    })
    
    fig = px.bar(df, 
                 y='classes', x='probabilidades (%)', orientation='h',
                 text=df['probabilidades (%)'].apply(lambda x: f'{x:.2f}%'), 
                 title='Confiança do Modelo no Diagnóstico', range_x=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

def main():
    # ==============================================================================
    # MUDANÇA 4: Apenas textos e títulos atualizados.
    st.set_page_config(
        page_title="Diagnóstico de Câncer Mamário",
        page_icon="🔬",
    )
    st.title("IA para Diagnóstico de Câncer Mamário 🔬")
    # ==============================================================================

    with st.spinner('Carregando modelo de IA...'):
        interpreter = carrega_modelo()
    
    image = carrega_imagem()
    
    if image is not None:
        previsao(interpreter, image)

if __name__ == "__main__":
    main()