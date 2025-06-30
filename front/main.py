import sys
import os
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_path = os.path.join(root_dir, "build")
datasets_dir = os.path.join(root_dir, "datasets")
os.makedirs(datasets_dir, exist_ok=True)
sys.path.append(build_path)
# --- END MODIFICATION ---

from py_arkade import ArkadeModel, FastRNN

import streamlit as st
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
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

@st.cache_data
def load_image_dataset(dataset_name):
    """Loads and caches a specified IMAGE dataset, resizing images to 64x64."""
    if dataset_name == "LFW People":
        data = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
        images_pil = [Image.fromarray(img).resize((64, 64), Image.Resampling.LANCZOS) for img in data.images]
        return np.array([np.array(img) for img in images_pil])
    elif dataset_name == "Olivetti Faces":
        data = fetch_olivetti_faces()
        return data.images
    return None

@st.cache_data
def load_point_cloud(uploaded_file):
    """Loads a 3D point cloud from an uploaded text file."""
    if uploaded_file is None:
        return None
    try:
        point_cloud = np.loadtxt(uploaded_file)
        if point_cloud.shape[1] != 3:
            st.error("Error: The file must contain points in 3D (three columns).")
            return None
        return point_cloud
    except Exception as e:
        st.error(f"Error reading or parsing file: {e}")
        return None
# --- END NEW FUNCTION ---


def get_imagen_gris(ruta, shape=(64, 64)):
    imagen = Image.open(ruta).convert("L").resize(shape)
    return np.array(imagen).reshape(-1)

def obtener_vecinos(imagen_path, dataset_original, model_type, k=5, distance_metric="2", search_radius=10.0, shape=(64, 64)):
    flat_data = dataset_original.reshape(dataset_original.shape[0], -1) if dataset_original.ndim == 3 else dataset_original
    query_vec = get_imagen_gris(imagen_path, shape=shape)

    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(flat_data)
    query_3d = pca.transform([query_vec])[0]

    data_path = os.path.join(datasets_dir, "dataset_3d.txt")
    np.savetxt(data_path, data_3d, fmt='%.6f')
    output_file = "./dataset_3d_results.txt"

    with open(output_file, 'w'): pass

    model_class = ArkadeModel if model_type == 'ArkadeModel' else FastRNN
    model_class(dataPath=data_path, distance=distance_metric, radio=search_radius, k=k,
                num_data_points=flat_data.shape[0], num_search=1, TrueKnn=True,
                outputFile=output_file, fromUser=True, myInput=query_3d)

    neighbors = []
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                neighbors.append((int(parts[0]), float(parts[1])))
    neighbors.sort(key=lambda x: x[1])
    return [idx for idx, _ in neighbors[:k]]

def run_point_cloud_test(point_cloud_data, model_type, k, distance_metric, search_radius):
    """
    Picks a random point from the cloud as a query, runs the model,
    and returns stats, query index, and neighbor indices.
    """
    data_3d = point_cloud_data

    query_idx = np.random.randint(0, data_3d.shape[0])
    query_3d = data_3d[query_idx]

    data_path = os.path.join(datasets_dir, "compare_point_cloud.txt")
    output_file = "./compare_point_cloud_results.txt"
    stats_file = output_file + "aux"

    np.savetxt(data_path, data_3d, fmt='%.6f')
    with open(output_file, 'w'): pass
    if os.path.exists(stats_file): os.remove(stats_file)

    model_class = ArkadeModel if model_type == 'ArkadeModel' else FastRNN
    model_class(dataPath=data_path, distance=distance_metric, radio=search_radius, k=k,
                num_data_points=data_3d.shape[0], num_search=1, TrueKnn=True,
                outputFile=output_file, fromUser=True, myInput=query_3d)

    try:
        with open(stats_file, 'r') as f:
            stats_content = f.read()
    except FileNotFoundError:
        stats_content = "Archivo de estadísticas no encontrado."

    neighbors = []
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                neighbors.append((int(parts[0]), float(parts[1])))
    neighbors.sort(key=lambda x: x[1])
    neighbor_indices = [idx for idx, _ in neighbors[:k]]

    return stats_content, query_idx, neighbor_indices

def plot_3d_neighbors(point_cloud, query_idx, neighbor_indices, title):
    """Generates an interactive 3D scatter plot of the point cloud and its neighbors."""
    df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])
    df['type'] = 'Point'
    # Ensure query is not overwritten if it's also a neighbor
    if query_idx in neighbor_indices:
        neighbor_indices.remove(query_idx)
    df.loc[neighbor_indices, 'type'] = 'Neighbor'
    df.loc[query_idx, 'type'] = 'Query'


    fig = px.scatter_3d(df, x='x', y='y', z='z', color='type', title=title,
                        color_discrete_map={
                            'Point': 'lightgrey',
                            'Query': 'red',
                            'Neighbor': '#00ffcc'
                        },
                        hover_data={'x': ':.3f', 'y': ':.3f', 'z': ':.3f', 'type': True})

    fig.update_traces(marker=dict(size=3, opacity=0.7), selector=dict(name='Point'))
    fig.update_traces(marker=dict(size=6, symbol='cross'), selector=dict(name='Neighbor'))
    fig.update_traces(marker=dict(size=8, symbol='diamond'), selector=dict(name='Query'))
    fig.update_layout(legend_title_text='Point Type')
    return fig

#--------INTERFAZ DE USUARIO--------

def subir_imagen(tam):
    poster = st.file_uploader("Suba un rostro", type=["jpg", "png", "jpeg"])
    if poster:
        content = poster.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            st.session_state.PATH = tmp.name
        img = Image.open(st.session_state.PATH)
        gray = img.convert("L").resize((64, 64))
        c1, _, c2 = st.columns([1, 0.2, 1])
        with c1: st.image(img, caption="Original", width=tam)
        with c2: st.image(gray, caption="Gris & redimensionada", width=tam)


def mostrar_vecinos(model_type, k, metric, radius):
    if "DATASET" not in st.session_state:
        with st.spinner("Cargando dataset de rostros LFW..."):
            st.session_state.DATASET = load_image_dataset("LFW People")

    imgs = st.session_state.DATASET
    neigh_ids = obtener_vecinos(st.session_state.PATH, imgs, model_type=model_type, k=k,
                                distance_metric=metric, search_radius=radius, shape=(64, 64))

    aux_file = "./dataset_3d_results.txtaux"
    st.subheader("Estadísticas de ejecución:")
    try:
        with open(aux_file, 'r') as af:
            st.code(af.read(), language='text')
    except FileNotFoundError:
        st.warning(f"Archivo de estadísticas no encontrado: {aux_file}")

    st.subheader(f"Vecinos más cercanos ({model_type}):")
    cols = st.columns(k if k <= 10 else 10)
    for i, idx in enumerate(neigh_ids):
        with cols[i % 10]:
            st.image(imgs[idx], width=100, caption=f"#{i+1}")


def main():
    st.set_page_config(page_title="aRKDe KNN Explorer", layout="wide")
    titulo_gud("Bienvenido a", "aRKDe")

    st.sidebar.title("Parámetros KNN")
    # model_type not needed here, selected in each column
    num_neighbors = st.sidebar.slider("Número de vecinos (k)", 1, 50, 5)
    distance_metric = st.sidebar.selectbox("Métrica de distancia", ["1", "2", "inf"], index=1,
                                           help="1: Manhattan; 2: Euclidiana; inf: Chebyshev")
    search_radius = st.sidebar.number_input("Radio de búsqueda", min_value=0.1, max_value=100.0,
                                            value=10.0, step=0.1)
    st.sidebar.markdown("---")
    st.sidebar.title("Acciones")
    seleccion = st.sidebar.selectbox("", ["Búsqueda de imágenes", "Comparativa de Nube de Puntos"])

    if seleccion == "Búsqueda de imágenes":
        st.header("Búsqueda de Vecinos por Imagen")
        model_type = st.sidebar.selectbox("Modelo a usar", ["ArkadeModel", "FastRNN"], index=0)
        subir_imagen(300)
        if st.sidebar.button("Obtener vecinos"):
            if "PATH" in st.session_state:
                mostrar_vecinos(model_type, num_neighbors, distance_metric, search_radius)
            else:
                st.warning("Por favor, suba primero una imagen.")

    else:
        st.header("Comparativa en Nube de Puntos 3D")
        st.write("""
            Suba un archivo de texto con una nube de puntos 3D (columnas x, y, z)
            para comparar el rendimiento de `ArkadeModel` y `FastRNN`. La prueba
            seleccionará un punto aleatorio del dataset, lo usará como consulta y
            visualizará los vecinos encontrados.
        """)

        uploaded_file = st.file_uploader(
            "Cargue su archivo de nube de puntos (.txt, .xyz, etc.)",
            type=['txt', 'xyz', 'dat']
        )

        if uploaded_file is not None:
            point_cloud = load_point_cloud(uploaded_file)

            if point_cloud is not None:
                st.success(f"Dataset cargado. Contiene {point_cloud.shape[0]} puntos.")

                col1, col2 = st.columns(2)

                # ArkadeModel Column
                with col1:
                    st.subheader("ArkadeModel")
                    if st.button("Ejecutar Prueba con ArkadeModel"):
                        with st.spinner("Ejecutando ArkadeModel..."):
                            stats, q_idx, n_indices = run_point_cloud_test(
                                point_cloud, 'ArkadeModel', num_neighbors,
                                distance_metric, search_radius
                            )
                            st.session_state.arkade_stats = stats
                            st.session_state.arkade_plot = plot_3d_neighbors(
                                point_cloud, q_idx, n_indices, 'ArkadeModel Results'
                            )

                    if 'arkade_plot' in st.session_state:
                        st.plotly_chart(st.session_state.arkade_plot, use_container_width=True)
                        st.write("Resultados para ArkadeModel:")
                        st.code(st.session_state.arkade_stats, language='text')

                # FastRNN Column
                with col2:
                    st.subheader("FastRNN")
                    if st.button("Ejecutar Prueba con FastRNN"):
                        with st.spinner("Ejecutando FastRNN..."):
                            stats, q_idx, n_indices = run_point_cloud_test(
                                point_cloud, 'FastRNN', num_neighbors,
                                distance_metric, search_radius
                            )
                            st.session_state.fastrnn_stats = stats
                            st.session_state.fastrnn_plot = plot_3d_neighbors(
                                point_cloud, q_idx, n_indices, 'FastRNN Results'
                            )

                    if 'fastrnn_plot' in st.session_state:
                        st.plotly_chart(st.session_state.fastrnn_plot, use_container_width=True)
                        st.write("Resultados para FastRNN:")
                        st.code(st.session_state.fastrnn_stats, language='text')

if __name__ == "__main__":
    main()