# CÓDIGO CORRIGIDO PARA SALVAR O MODELO (MAIS COMPATÍVEL)

import tensorflow as tf
import os
from google.colab import drive

print("--- INICIANDO PROCESSO DE CONVERSÃO E SALVAMENTO (VERSÃO COMPATÍVEL) ---")

try:
    # Garante que o modelo existe na memória
    if 'model' not in locals() and 'model' not in globals():
        raise NameError("A variável 'model' com o modelo treinado não foi encontrada.")
    print("✅ Modelo treinado ('model') encontrado na memória.")
    
    # Monta o Drive e cria a pasta de destino
    drive.mount('/content/drive', force_remount=True)
    SAVE_DIR = '/content/drive/MyDrive/TCC/testeTCC-modelos-treinados/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("✅ Drive e pasta de destino prontos.")

    # --- INÍCIO DA CORREÇÃO ---
    # Caminho para o novo modelo
    TFLITE_MODEL_PATH = os.path.join(SAVE_DIR, 'modelo_compativel_quantizado.tflite')
    print(f"\nO novo modelo será salvo em: {TFLITE_MODEL_PATH}")

    # Inicializa o conversor a partir do nosso modelo Keras
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # APLICA A OTIMIZAÇÃO DE FAIXA DINÂMICA (MAIS COMPATÍVEL)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Removemos a linha "target_spec" que causava o problema de compatibilidade
    # converter.target_spec.supported_types = [tf.float16]
    
    # --- FIM DA CORREÇÃO ---

    # Executa a conversão
    modelo_tflite_quantizado = converter.convert()
    print("✅ Modelo convertido com sucesso.")

    # Salva o novo arquivo
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(modelo_tflite_quantizado)
        
    print(f"\n🎉 SUCESSO! Novo modelo salvo com sucesso.")

    # Verificação final
    print("\nVerificando o arquivo na pasta de destino:")
    !ls -lh "{SAVE_DIR}"

except Exception as e:
    print(f"❌ Ocorreu um erro durante o processo: {e}")