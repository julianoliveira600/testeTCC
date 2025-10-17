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
    
    #https://drive.google.com/file/d/1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1H1fcJRSzEMIpX5gidh6Z32Uo9owO-u5d'

    gdown.dowload(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte sua imagem ou clique para selecionar uma', type=['png','jpg', 'jpg', 'jpeg'])
    if uploaded_file is not None: 
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Imagem carregada com sucesso')

        image = np.array(image, dtype=np.float32)
        image = image.resize((224, 224))
        image = np.expand_dims(image, axis)

        return image

def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['benign', 'malign']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df, y='classes', x = 'probabilidades (%)', orientation='h', text='probabilidades (%)', title='Probabilidade de Tipo de Tipo de Cancer')
    st.plotly_chart(fig)
def main():
    st.set_page_config(
        page_title= "Classifica Imagens Cancer Mamario em Animais Domesticos",
        page_icon="=>",
    )

    st.write("Classifica Imagens Cancer Mamario em Animais Domesticos")

    #Carrega modelo
    interpreter = carrega_modelo
    #Carrega Imagem
    image = carrega_imagem()
    #Classifica
    if image is not None:
        previsao(interpreter, image)


if __name__=="__main__":
    main()