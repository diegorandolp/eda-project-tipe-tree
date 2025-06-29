import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import tempfile

# ---------ESTILOS----------
def titulo_gud(y, x):
    st.markdown(f"""
        <h1 style='
            font-family: monospace;
            font-size: 50px;
            text-align: center;
            margin-top: 30px;
            color: white;
        '>
        {y} <span style='color: #00ffcc; '>{x}</span>
        </h1>
    """, unsafe_allow_html=True)
# ---------FIN -------------

#-----------PROCESAMIENTO-----------
def cargar_imagen_gris(ruta, shape=(64, 64)):
    imagen = Image.open(ruta).convert("L")  
    imagen_redimensionada = imagen.resize(shape)
    return np.array(imagen_redimensionada)

def procesar_imagen_gris(imagen_poster, shape=(64, 64)):
    imagen = Image.open(imagen_poster).convert("L").resize(shape)
    return imagen

def get_imagen_gris(ruta, shape=(64, 64)):
    imagen = Image.open(ruta).convert("L").resize(shape)
    return np.array(imagen).reshape(-1) 

# Pruebas aún
def obtener_vecinos(imagen_path, dataset_original, k=4, shape=(64, 64)):
    if dataset_original.ndim == 3:
        dataset_original = dataset_original.reshape((dataset_original.shape[0], -1))
    imagen_aplanada = get_imagen_gris(imagen_path, shape=shape)

    pca = PCA(n_components=3)
    dataset_3d = pca.fit_transform(dataset_original)
    imagen_3d = pca.transform([imagen_aplanada])[0]

    similitudes = np.dot(dataset_3d, imagen_3d)
    indices_top_k = np.argsort(similitudes)[-k:][::-1]
    return indices_top_k

def recover_nearest(k_nearest, dataset_original):
    return dataset_original[k_nearest]


#----------FIN PROCESAMIENTO --------



#--------INTERFAZ DE USUARIO--------
DATASET_CARGADO = None
PATH = None

def subirImagen(tam):
    poster = st.file_uploader("Suba un rostro", type=["jpg", "png", "jpeg"])
    if poster is not None:
        contenido = poster.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(contenido)
            st.session_state.PATH =tmp_file.name
            st.text(f"Imagen guardada en: {st.session_state.PATH}")

        imagen = Image.open(tmp_file.name)
        img_gris = imagen.convert("L").resize((64, 64))

        col1, centro, col2 = st.columns([1, 0.2, 1])
        with col1:
            st.image(imagen, caption="imagen original", width=tam)
        with col2:
            st.image(img_gris, caption="imagen en gris y redimensionada", width=tam)



def plotear_top_k(path ,k_=9):
    if "DATASET_CARGADO" not in st.session_state:
        st.session_state.DATASET_CARGADO = fetch_lfw_people(min_faces_per_person=0, resize=1)

    data = st.session_state.DATASET_CARGADO
    images = data.images 
    labels = data.target_names[data.target]

    k_nearest = obtener_vecinos(path, images, k=k_, shape=(125, 94))          
    imagenes = recover_nearest(k_nearest, images)
    columns_ = 4
    for i in range(0, len(imagenes), columns_):
        cols = st.columns(columns_)
        for j in range(columns_):
            if i + j < len(imagenes):
                with cols[j]:
                    st.image(imagenes[i + j], width=150, caption=f"Vecino #{i + j + 1}")

if "DATASET_CARGADO" not in st.session_state:
    with st.spinner("Cargando dataset de rostros..."):
        st.session_state.DATASET_CARGADO = fetch_lfw_people(min_faces_per_person=0, resize=1)

def main():
    titulo_gud("Bienvenido a", "aRKDe")
    
    # Menú lateral
    st.sidebar.title("Menu")
    seleccion = st.sidebar.selectbox(
        "",
        ["Búsqueda de imágenes", "Comparativa"]
    )
    if seleccion == "Búsqueda de imágenes":
        subirImagen(300)
        if st.button("Obtener vecinos"):
            plotear_top_k(path=st.session_state.PATH,k_=20)



if __name__ == "__main__":
    main()