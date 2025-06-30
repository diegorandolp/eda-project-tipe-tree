import sys
import os

# Get the absolute path to the project's root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_path = os.path.join(root_dir, "build")
sys.path.append(build_path)

try:
    from py_arkade import ArkadeModel

    data_path = "../datasets/sample_gowalla.txt"
    distance_metric = "2"
    search_radius = 10.0
    num_neighbors = 5
    total_data_points = 1000
    query_points = 10
    with_true_knn = True
    # i had to create it manually
    output_file = "./knn_results9.txt"

    print("Creating ArkadeModel instance...")
    model = ArkadeModel(
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

