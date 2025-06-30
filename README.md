# Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing - Reproduction Work
This project is a reproduction of the source code described in the paper:
https://doi.org/10.1145/3650200.3656601

## Team members

- Daniel Casquino
- Josep Castro
- Marcelino Maita
- Rodrigo Meza
- Diego Quispe

## Prerequisites
Before you begin, ensure you have the following installed:

OptiX: A high-performance ray tracing engine from NVIDIA.

Intel TBB (Threading Building Blocks): A C++ template library for parallel programming.

## Installation Steps
Follow these steps to install the necessary components and build the project:

1. Install OptiX and Intel TBB:
Ensure that OptiX and Intel TBB are installed on your system. The specific installation method may vary depending on your operating system and NVIDIA driver version. Refer to their official documentation for detailed instructions.

2. Build the Project:
Navigate to your project's root directory and execute the following commands:

```bash
mkdir build
cd build
cmake ../ -DKN=3 -DNORM=2
make s01-knn
```


- ```mkdir build```: Creates a new directory named build.

- ```cd build```: Changes the current directory to build.

- ```cmake ../```: Configures the project using CMake.

- ```make```: Compiles the Arkade project and creates the bindings.

## Running the Application
After successfully building the project, you can run the Streamlit application:

1. Set ```LD_PRELOAD``` (if necessary):
If you encounter issues related to libstdc++.so.6, you might need to preload the library:
```bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
```

2. Start the Streamlit Application:
Navigate to the front directory and start the Streamlit application:

```bash
cd front
streamlit start main.py
```
This command will launch the Streamlit application in your web browser.


## Publication

Please cite this paper if you find Arkade useful.

```bib
@inproceedings{10.1145/3650200.3656601,
author = {Mandarapu, Durga Keerthi and Nagarajan, Vani and Pelenitsyn, Artem and Kulkarni, Milind},
title = {Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing},
year = {2024},
isbn = {9798400706103},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3650200.3656601},
doi = {10.1145/3650200.3656601},
booktitle = {Proceedings of the 38th ACM International Conference on Supercomputing},
pages = {14–25},
numpages = {12},
keywords = {GPU Ray Tracing, Non-Euclidean Distances, k-Nearest Neighbor Search},
location = {Kyoto, Japan},
series = {ICS '24}
}
```
