#include "ArkadeModel.h"
#include "CreateBVH.h"
#include <utility>

bool is_open(const string& df, const string& of) {
    ifstream file1(df);
    ifstream file2(of);

    return file1.is_open() && file2.is_open();
}

ArkadeModel::ArkadeModel(std::string dataPath, string _distance, float _radio, int _k, int _num_data_points,
                         int _num_search, std::string outputPath) {

    dataFile = std::move(dataPath);
    distance = std::move(_distance);
    radio = _radio;
    k = _k;
    num_data_points = _num_data_points;
    num_search = _num_search;
    outputFile = std::move(outputPath);

    if(!is_open(dataFile, outputFile)) {
        cerr << "Error al abrir algún path mandado\n";
        exit(EXIT_FAILURE);
    }

    // Devolverá el mismo valor si es que no necesita una transformación monotoma (L^p)
    cout << "\npepepe: "<< distance;
    func = new TransMonotoma(distance);

    gpu_process = new CreateBVH(this);
    results = gpu_process->get_results();

    // to write the results to the output file
    cout << *this;
}

void ArkadeModel::InitializeData(){

    // Open DataSet
    ifstream file;
    file.open(dataFile);

    // Cantidad total de datos en el DataSet.
    int TotalData = 0;
    string line;


    // Rescatamos la información del dataset y la convertimos en Points, distribuido en la Data y Query.
    while(getline(file, line)){

        stringstream ss(line);
        float dim_val;
        vector<float> cords;

        while(ss >> dim_val){
            cords.push_back(dim_val);

            // Se espera la información separada por comas.
            if(ss.peek() == ',')
                ss.ignore();

        }

        EDA::Point pt;

        for(int idx = 0; idx < cords.size(); ++idx)
            pt.set_point(idx, cords[idx]);

        pt.pt = func->transformar(pt.pt);

        if(TotalData < num_data_points)
            DataPoints.push_back(pt);
        else
            QueryPoints.push_back(pt);


        TotalData+=1;

    }

    file.close();

    if(TotalData < num_data_points + num_search)
        throw runtime_error("No hay suficientes data para la busqueda.");


    // Inicializamos los k vecinos.
    for(size_t idx = 0; idx < k; ++idx){
        EDA::Neigh vecino{};
        vecino.dist = FLOAT_MIN;
        vecino.idx = -1;

        Vecinos.push_back(vecino);
    }

}

float ArkadeModel::get_radius() const {
    return radio;
}

int ArkadeModel::get_k() const {
    return k;
}

int ArkadeModel::get_num_search() const {
    return num_search;
}

string& ArkadeModel::get_distance_type() {
    return distance;
}
