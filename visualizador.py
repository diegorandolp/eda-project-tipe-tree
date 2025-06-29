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
import sys
import plotly.graph_objects as go
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

def obtener_vecinos_images(output_path):
    df = pd.read_csv(output_path, sep="\t", header=None, names=["fila", "valor"])
    df_ordenado = df.sort_values(by="valor", ascending=True)
    return df_ordenado["fila"].to_numpy()

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

def eliminar_ultima_fila(path_txt):
    with open(path_txt, "r") as f:
        lineas = f.readlines()
    if not lineas:
        return 
    lineas = lineas[:-1]
    with open(path_txt, "w") as f:
        f.writelines(lineas)


def plotear_top_k(path ,k_=9):
    # if "DATASET_CARGADO" not in st.session_state:
    #     st.session_state.DATASET_CARGADO = fetch_lfw_people(min_faces_per_person=0, resize=1)

    data = st.session_state.DATASET_CARGADO
    images = data.images 
    labels = data.target_names[data.target]

    ############ Caso especifico del dataset imagenes ############
    dataset_path = "dataset_lfw_people.txt"
    distance_metric = "2"
    num_data_points = 13234
    output_path = "lfw_people_results.txt"
    k_neighbors = 10
    num_search = 1
    # Insertar al ultimo dataset_path la imagen que ta en el path aplciandole PCA

    # ----- Preparación de la imágen para insertarlo -----
    imagen_aplanada = get_imagen_gris(path, shape=(125, 94))
    pca = PCA(n_components=3)
    pca = st.session_state.PCA_3D
    imagen_3d = pca.transform([imagen_aplanada])[0]

    print(imagen_3d)
    with open(dataset_path, "a") as f:
        fila_txt = " ".join(map(str, imagen_3d.tolist()))
        f.write(fila_txt + "\n")

    instanciar_Arkade_model(dataset_path, distance_metric, num_data_points, output_path, k_neighbors, num_search, query_points=1)
    # k_nearest = obtener_vecinos(path, images, k=k_, shape=(125, 94))          
    k_nearest = obtener_vecinos_images(output_path)
    # Eliminar la ultima fila: 
    eliminar_ultima_fila(dataset_path)
    
    imagenes = recover_nearest(k_nearest, images)
    columns_ = 4
    for i in range(0, len(imagenes), columns_):
        cols = st.columns(columns_)
        for j in range(columns_):
            if i + j < len(imagenes):
                with cols[j]:
                    st.image(imagenes[i + j], width=150, caption=f"Vecino #{i + j + 1}")

if "DATASET_CARGADO" not in st.session_state:
    with st.spinner("Cargando dataset de rostros y PCA..."):
        st.session_state.DATASET_CARGADO = fetch_lfw_people(min_faces_per_person=0, resize=1)
        st.session_state.PCA_3D = PCA(n_components=3).fit(st.session_state.DATASET_CARGADO.data)


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

def plot_3d(df, query_point, title="Distribución 3D"):
    df_plot = df.copy()
    df_plot.columns = ['x', 'y', 'z']
    # print(df_plot.shape)
    # print(df_plot.index.is_unique) 
    # print(df.describe())
    # print(df.isnull().sum())
    df_plot['title'] = df_plot.index.astype(str)

    df_plot['legend'] = 'datapoints'
    if query_point > 0:
        df_plot.iloc[-query_point:, df_plot.columns.get_loc('legend')] = 'query points'

    # scaler = StandardScaler()
    # df_plot[['x', 'y', 'z']] = scaler.fit_transform(df_plot[['x', 'y', 'z']])
    # df_plot[['x', 'y', 'z']] *= 10000
    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z',
        color='legend',
        hover_name='title',
        title=title,
        opacity=0.7, 
        # color_discrete_sequence=px.colors.qualitative.Safe
        # color_discrete_sequence=px.colors.qualitative.Plotly
        # color_discrete_sequence=px.colors.qualitative.Vivid
        color_discrete_map={
            'datapoints': 'blue',
            'query points': 'red'
        }

    )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # fig.add_annotation(text=f"Total puntos: {len(df_plot)}", showarrow=False, xref='paper', yref='paper', x=0, y=1.1)
    st.plotly_chart(fig, use_container_width=True)

def start_params():
    col1_seleccion, col2_seleccion = st.columns(2)

    with col1_seleccion:
        data_path = st.text_input("Coloque la ruta del dataset (.txt)", value="gowalla_loc.txt")
        distance_metric = st.text_input("Distancia", value="2")
        total_data_points = st.number_input("Total de puntos", min_value=1, max_value=4000, value=1000, step=1)
    with col2_seleccion:
        output_path = st.text_input("Coloque la ruta de resultados (.txt)", value="KnnResults.txt")
        search_radius = st.number_input("Ingresa el radio", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        num_neighbors = st.number_input("k", min_value=1, max_value=4000, value=5, step=1)
        query_points = st.number_input("Puntos de búsqueda", min_value=1, max_value=4000, value=1, step=1)
    return data_path, distance_metric, total_data_points, output_path, search_radius, num_neighbors, query_points

def instanciar_Arkade_model(data_path, distance_metric, total_data_points,
                      output_path, search_radius, num_neighbors, query_points):
    try:
        from py_arkade import ArkadeModel
        print("Creating ArkadeModel instance...")
        model = ArkadeModel(
            dataPath=data_path,
            distance=distance_metric,
            radio=search_radius,
            k=num_neighbors,
            num_data_points=total_data_points,
            num_search=query_points,
            outputFile=output_path
        )
        print("ArkadeModel instance created successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to instantiate ArkadeModel: {e}")
        return None

def openTiempos(output_path):
    tiempos_path = output_path.replace('.txt', '_tiempos.txt')
    try:
        with open(tiempos_path, 'r') as f:
            line = f.readline().strip()
        valores = [float(v.strip()) for v in line.split()]
        return valores

    except Exception as e:
        print(f"Error reading tiempos file: {e}")
        return []
    

def graficar_tiempos(tiempos_, title, log_scale=False):
    tiempos = tiempos_.copy()[:2]
    tiempos.append(tiempos[0]+tiempos[1])

    labels = [
        "Construcción BVH",
        "Filter & Refine", 
        "Tiempo total"
    ]

    fig = go.Figure(data=[
        go.Bar(x=labels, y=tiempos, marker_color='indianred')
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Etapas",
        yaxis_title="Tiempo (escala logarítmica)" if log_scale else "Tiempo",
        template="plotly_white", 
        yaxis_type="log" if log_scale else "linear"
    )

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

    elif seleccion == "Comparativa":
        data_path, distance_metric, total_data_points, output_path, search_radius, num_neighbors, query_points = start_params()
        
        arkade_model = instanciar_Arkade_model(data_path, distance_metric, total_data_points,
                      output_path, search_radius, num_neighbors, query_points)

        # if arkade_model is not None:
        print("Leyendo archivos para mostrar resultados.")
        df_dataset = open_dataset(data_path) 
        df_knn_results = open_outputPath(output_path) 
        df_fila = take_fil( df_dataset, df_knn_results) 
        plot_3d(df_fila,  query_points, "Resultados")
        
        # tiempos = output_path -".txt" + "_tiempos.txt"
        # una sola linea de: 
        # contruccion BVH, Filter & Refine, Numero de intersecciones, Total de valores a escribir:

        tiempos = openTiempos(output_path)
        if tiempos:
            print(tiempos)
            graficar_tiempos(tiempos,"Tiempos de arkade" , False)
        else:
            print("Tiempos esta vacío")





if __name__ == "__main__":
    main()