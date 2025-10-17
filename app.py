# app.py - VERSÃO FINAL E COMPLETA

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
    # ==============================================================================
    # !! IMPORTANTE !!
    # COLE AQUI O NOVO LINK DE COMPARTILHAMENTO GERADO NO PASSO 1
    # Link para o arquivo: 'modelo_final_compativel.tflite'
    #url = 'https://drive.google.com/uc?id=1Gp3W6VwF4CpwQEmbAOuVW4mopXluVex9'
    #https://drive.google.com/file/d/1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d' 
    # ==============================================================================
    
    output_filename = 'modelo_final_compativel.tflite'

    # Baixa o arquivo do Google Drive se ele ainda não existir localmente
    gdown.download(url, output_filename, quiet=False)
    
    # Carrega o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=output_filename)
    interpreter.allocate_tensors()
    
    return interpreter

def carrega_e_prepara_imagem():
    """
    Cria a interface de upload e pré-processa a imagem para o formato que o modelo espera.
    """
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Lê e abre a imagem
        image_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_data))

        st.image(pil_image, caption="Imagem Carregada", use_column_width=True)
        st.success('Imagem carregada com sucesso!')

        # Pré-processamento: Redimensiona para 224x224 e converte para o formato correto
        pil_image_resized = pil_image.resize((224, 224))
        image_array = np.array(pil_image_resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    return None

def faz_previsao(interpreter, image_array):
    """
    Recebe o modelo e a imagem preparada, executa a predição e mostra os resultados.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    score = output_data[0][0]
    
    if score < 0.5:
        st.write("## Diagnóstico: **Benigno**")
    else:
        st.write("## Diagnóstico: **Maligno**")

    prob_benigno = 100 * (1 - score)
    prob_maligno = 100 * score
    
    df = pd.DataFrame({
        'Classe': ['Benigno', 'Maligno'],
        'Probabilidade (%)': [prob_benigno, prob_maligno]
    })
    
    fig = px.bar(df,
                 y='Classe', x='Probabilidade (%)', orientation='h',
                 text=df['Probabilidade (%)'].apply(lambda x: f'{x:.2f}%'),
                 title='Confiança do Modelo no Diagnóstico', range_x=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)

# --- FUNÇÃO PRINCIPAL ---
def main():
    """
    Função principal que organiza e executa o aplicativo Streamlit.
    """
    st.set_page_config(
        page_title="Sistema de Diagnóstico de Câncer",
        page_icon="🔬",
        layout="centered"
    )
    
    st.title("Sistema de Diagnóstico de Câncer por IA 🔬")
    st.write("Faça o upload de uma imagem histopatológica para que o modelo a classifique como benigna ou maligna.")

    try:
        with st.spinner('Carregando modelo, por favor aguarde...'):
            interpreter = carrega_modelo()
        
        st.success("Modelo carregado com sucesso!")

        image_array = carrega_e_prepara_imagem()
        
        if image_array is not None:
            faz_previsao(interpreter, image_array)
            
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        st.error("Verifique se o link de compartilhamento do Google Drive está correto e com a permissão 'Qualquer pessoa com o link'.")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()