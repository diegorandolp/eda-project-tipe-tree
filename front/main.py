import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_path = os.path.join(root_dir, "build")
sys.path.append(build_path)
from py_arkade import ArkadeModel, FastRNN

import streamlit as st
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import tempfile
import plotly.express as px

# ---------ESTILOS----------
def titulo_gud(y, x):
    st.markdown(f"""
        <h1 style='\
            font-family: monospace;\
            font-size: 50px;\
            text-align: center;\
            margin-top: 30px;\
            color: white;\
        '>
        {y} <span style='color: #00ffcc; '>{x}</span>
        </h1>
    """, unsafe_allow_html=True)
# ---------FIN -------------

#-----------PROCESAMIENTO-----------
def get_imagen_gris(ruta, shape=(64, 64)):
    imagen = Image.open(ruta).convert("L").resize(shape)
    return np.array(imagen).reshape(-1)

def obtener_vecinos(imagen_path, dataset_original, model_type, k=5, distance_metric="2", search_radius=10.0, shape=(64, 64)):
    flat_data = dataset_original.reshape(dataset_original.shape[0], -1) if dataset_original.ndim == 3 else dataset_original
    # Prepare query vector
    query_vec = get_imagen_gris(imagen_path, shape=shape)

    # PCA to 3D
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(flat_data)
    query_3d = pca.transform([query_vec])[0]

    # Arkade parameters
    data_path = "../datasets/dataset_3d.txt"
    output_file = "./dataset_3d_results.txt"

    with open(output_file, 'w'): pass

    if model_type == 'ArkadeModel':
        model = ArkadeModel(
            dataPath=data_path,
            distance=distance_metric,
            radio=search_radius,
            k=k,
            num_data_points=flat_data.shape[0],
            num_search=1,
            TrueKnn=True,
            outputFile=output_file,
            fromUser=True,
            myInput=query_3d
        )
    else:
        model = FastRNN(
            dataPath=data_path,
            distance=distance_metric,
            radio=search_radius,
            k=k,
            num_data_points=flat_data.shape[0],
            num_search=1,
            TrueKnn=True,
            outputFile=output_file,
            fromUser=True,
            myInput=query_3d
        )

    neighbors = []
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                neighbors.append((int(parts[0]), float(parts[1])))
    neighbors.sort(key=lambda x: x[1])
    return [idx for idx, _ in neighbors[:k]]
#----------FIN PROCESAMIENTO --------

#--------INTERFAZ DE USUARIO--------

def subir_imagen(tam):
    poster = st.file_uploader("Suba un rostro", type=["jpg", "png", "jpeg"])
    if poster:
        content = poster.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            st.session_state.PATH = tmp.name
            st.text(f"Imagen guardada en: {st.session_state.PATH}")
        img = Image.open(st.session_state.PATH)
        gray = img.convert("L").resize((64, 64))
        c1, _, c2 = st.columns([1, 0.2, 1])
        with c1:
            st.image(img, caption="Original", width=tam)
        with c2:
            st.image(gray, caption="Gris & redimensionada", width=tam)


def mostrar_vecinos(model_type, k, metric, radius):
    if "DATASET" not in st.session_state:
        with st.spinner("Cargando dataset de rostros..."):
            st.session_state.DATASET = fetch_lfw_people(min_faces_per_person=0, resize=1)

    imgs = st.session_state.DATASET.images
    neigh_ids = obtener_vecinos(
        st.session_state.PATH,
        imgs,
        model_type=model_type,
        k=k,
        distance_metric=metric,
        search_radius=radius,
        shape=(125, 94)
    )

    aux_file = "./dataset_3d_results.txtaux"
    st.subheader("Estadísticas de ejecución:")
    try:
        with open(aux_file, 'r') as af:
            for line in af:
                st.text(line.strip())
    except FileNotFoundError:
        st.warning(f"Archivo de estadísticas no encontrado: {aux_file}")

    st.subheader(f"Vecinos más cercanos ({model_type}):")
    cols = st.columns(4)
    for i, idx in enumerate(neigh_ids):
        with cols[i % 4]:
            st.image(imgs[idx], width=150, caption=f"#{i+1}")


def main():
    st.set_page_config(page_title="aRKDe KNN Explorer", layout="wide")
    titulo_gud("Bienvenido a", "aRKDe")

    st.sidebar.title("Parámetros KNN")
    model_type = st.sidebar.selectbox("Modelo a usar", ["ArkadeModel", "FastRNN"], index=0)
    num_neighbors = st.sidebar.slider("Número de vecinos (k)", 1, 50, 5)
    distance_metric = st.sidebar.selectbox(
        "Métrica de distancia", ["1", "2", "inf"], index=1,
        help="1: Manhattan; 2: Euclidiana; inf: Chebyshev"
    )
    search_radius = st.sidebar.number_input(
        "Radio de búsqueda", min_value=0.1, max_value=100.0,
        value=10.0, step=0.1
    )
    st.sidebar.markdown("---")
    st.sidebar.title("Acciones")
    seleccion = st.sidebar.selectbox("", ["Búsqueda de imágenes", "Comparativa"])

    if seleccion == "Búsqueda de imágenes":
        subir_imagen(300)
        if st.sidebar.button("Obtener vecinos"):
            if "PATH" in st.session_state:
                mostrar_vecinos(model_type, num_neighbors, distance_metric, search_radius)
            else:
                st.warning("Por favor, suba primero una imagen.")
    else:
        st.info("En construcción: Vista comparativa aún no disponible.")

if __name__ == "__main__":
    main()
