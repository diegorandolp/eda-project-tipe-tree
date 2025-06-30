import sys
import os

# Get the absolute path to the project's root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_path = os.path.join(root_dir, "build")
sys.path.append(build_path)
from py_arkade import ArkadeModel

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import tempfile
from sklearn.preprocessing import StandardScaler
import plotly.express as px

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
def obtener_vecinos(imagen_path,dataset_original, k=4, shape=(64, 64)):
    if dataset_original.ndim == 3:
        dataset_original = dataset_original.reshape((dataset_original.shape[0], -1))
    imagen_aplanada = get_imagen_gris(imagen_path, shape=shape)

    pca = PCA(n_components=3)
    dataset_3d = pca.fit_transform(dataset_original)
    imagen_3d = pca.transform([imagen_aplanada])[0]

    data_path = "../datasets/dataset_3d.txt"
    distance_metric = "2"
    search_radius = 10.0
    num_neighbors = 5
    total_data_points = 13000
    query_points = 1
    # i had to create it manually
    output_file = "./dataset_3d_results.txt"
    from_user = True
    my_input = imagen_3d

    print("Creating ArkadeModel instance...")
    model = ArkadeModel(
        dataPath=data_path,
        distance=distance_metric,
        radio=search_radius,
        k=num_neighbors,
        num_data_points=total_data_points,
        num_search=query_points,
        TrueKnn=True,
        outputFile=output_file,
        fromUser=from_user,
        myInput=my_input)
    print("ArkadeModel instance created successfully.")

    with open(output_file, 'r') as file:
        lines = file.readlines()

    indices_top_k = []
    # sort distances
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        parts[0] = int(parts[0])
        parts[1] = float(parts[1])
        indices_top_k.append((parts[0], parts[1]))
    indices_top_k.sort(key=lambda x: x[1])  # Sort by distance

    return [i for i, _ in indices_top_k[:k]]

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

#if "DATASET_CARGADO" not in st.session_state:
#    with st.spinner("Cargando dataset de rostros..."):
#        st.session_state.DATASET_CARGADO = fetch_lfw_people(min_faces_per_person=0, resize=1)



# functions_f.ArkadeModel model(dataPath, distance, radio, k, num_data_points, num_search, outputPath);

def open_dataset(dataset_path):
    data = pd.read_csv(dataset_path, sep="\t", header=None)
    return data

def open_outputPath(outputPath):
    df = pd.read_csv(outputPath, sep="\t", header=None, names=["fila", "valor"])
    return df


def take_fil(df_data, df_res):
    selected = df_data.iloc[df_res["fila"].values]
    return selected

def plot_3d(df, title="Distribución 3D"):
    df_plot = df.copy()
    df_plot.columns = ['x', 'y', 'z']
    # print(df_plot.shape)
    # print(df_plot.index.is_unique)
    # print(df.describe())
    # print(df.isnull().sum())
    df_plot['title'] = df_plot.index.astype(str)

    df_plot['cluster'] = 'todos'
    # scaler = StandardScaler()
    # df_plot[['x', 'y', 'z']] = scaler.fit_transform(df_plot[['x', 'y', 'z']])
    # df_plot[['x', 'y', 'z']] *= 10000
    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z',
        color='cluster',
        hover_name='title',
        title=title,
        opacity=0.7,
        # color_discrete_sequence=px.colors.qualitative.Safe
        color_discrete_sequence=px.colors.qualitative.Plotly
        # color_discrete_sequence=px.colors.qualitative.Vivid

    )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # fig.add_annotation(text=f"Total puntos: {len(df_plot)}", showarrow=False, xref='paper', yref='paper', x=0, y=1.1)
    st.plotly_chart(fig, use_container_width=True)

def main():

    titulo_gud("Bienvenido a", "aRKDe")

    # Menú lateral
    st.sidebar.title("Menu")
    seleccion = st.sidebar.selectbox(
        "-----",
        ["Búsqueda de imágenes", "Comparativa"]
    )
    if seleccion == "Búsqueda de imágenes":
        subirImagen(300)
        if st.button("Obtener vecinos"):
            plotear_top_k(path=st.session_state.PATH,k_=20)
            print("Obteniendo vecinos...")
    elif seleccion == "Comparativa":
        data_path = st.text_input("Coloque la ruta del dataset (.txt)", value=data_path)
        output_path  = st.text_input("Coloque la ruta de resultados (.txt)", value=output_file)

        df_dataset = open_dataset(data_path)
        df_knn_results = open_outputPath(output_path)
        df_fila = take_fil( df_dataset, df_knn_results)
        # Plotear
        plot_3d(df_fila, "Resultados")




if __name__ == "__main__":
    main()