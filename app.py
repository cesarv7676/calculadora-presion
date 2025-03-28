import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU

# Cargar modelo y transformadores
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("/content/Modelo10variablesfinal.h5", custom_objects={'LeakyReLU': LeakyReLU})
    num_transformer = joblib.load("/content/num_transformer10final.pkl")
    onehot = joblib.load("/content/onehot10final.pkl")
    return model, num_transformer, onehot

model, num_transformer, onehot = load_resources()

# Definir columnas
num_cols = ['PASpre', 'PADpre', 'Cambio_peso', 'Ult_edad', 
            'HTA_antiguedad', 'Conteo_NUTRi_post', 'Prom_COLT_previo', 'Prom_peso_previo']
cat_cols = ['SEXOrev', 'DM_', 'ACT_Fisica']

# Interfaz de usuario
st.title("Calculadora de Control de Presión Arterial")

# Entradas numéricas
paspre = st.number_input("PASpre (mmHg):", value=180)
padpre = st.number_input("PADpre (mmHg):", value=97)
cambio_peso = st.number_input("Cambio de peso (kg):", value=-5.0)
ult_edad = st.number_input("Edad:", value=60)
hta_antiguedad = st.number_input("Antigüedad de HTA (años):", value=4)
conteo_nutri = st.number_input("Conteo NutRi post:", value=2)
prom_colt = st.number_input("Promedio COLT previo:", value=200)
prom_peso = st.number_input("Promedio peso previo (kg):", value=100)

# Entradas categóricas
sexo = st.selectbox("Sexo:", options=[0, 1], index=0)
dm = st.selectbox("DML:", options=[0, 1], index=0)
act_fisica = st.selectbox("Actividad Física:", options=[0, 1], index=0)

# Preprocesar y predecir
if st.button("Calcular Probabilidad"):
    input_data = {
        'PASpre': paspre,
        'PADpre': padpre,
        'Cambio_peso': cambio_peso,
        'Ult_edad': ult_edad,
        'HTA_antiguedad': hta_antiguedad,
        'Conteo_NUTRi_post': conteo_nutri,
        'Prom_COLT_previo': prom_colt,
        'Prom_peso_previo': prom_peso,
        'SEXOrev': sexo,
        'DM_': dm,
        'ACT_Fisica': act_fisica
    }
    
    X_num = num_transformer.transform(pd.DataFrame([input_data])[num_cols]
    X_cat = onehot.transform(pd.DataFrame([input_data])[cat_cols]
    X_processed = np.hstack([X_num, X_cat])
    
    prob = model.predict(X_processed, verbose=0)[0][0]
    st.success(f"**Probabilidad de control:** {prob:.2%}")