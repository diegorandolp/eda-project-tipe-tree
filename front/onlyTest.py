import sys
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_path = os.path.join(root_dir, "build")
sys.path.append(build_path)

try:
    from py_arkade import ArkadeModel, FastRNN

    data_path = "../datasets/sample_gowalla.txt"
    distance_metric = "5"
    num_neighbors = 5
    total_data_points = 1000
    query_points = 10
    with_true_knn = True
    # i had to create it manually
    output_file = "./knn_results9.txt"

    print("Estimating search radius...")
    df = pd.read_csv(data_path, sep='\t', header=None, nrows=total_data_points)
    list1 = df.values.tolist()

    samples = sample(list1,100)

    X = np.array(samples)

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric='l1').fit(X)

    distances, indices = nbrs.kneighbors(X)

    search_radius = min(distances, key=lambda x: x[1])[1]
    print(f"Estimated search radius = {search_radius}")

    print("Creating ArkadeModel instance...")

    model = FastRNN(
    #model = ArkadeModel(
        dataPath=data_path,
        distance=distance_metric,
        radio=search_radius,
        k=num_neighbors,
        num_data_points=total_data_points,
        num_search=query_points,
        TrueKnn=with_true_knn,
        outputFile=output_file)
    print("ArkadeModel instance created successfully.")



except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)
