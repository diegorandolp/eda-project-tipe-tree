#include "BaseModel.h"

#include <utility>

bool is_open(const string& df, const string& of) {
    ifstream file1(df);
    ifstream file2(of);

    return file1.is_open() && file2.is_open();
}
BaseModel::BaseModel(string dataPath, float _radio, int _k, int _num_data_points,
                     int _num_search, string outputPath, bool _fromUser, vector<float> _myInput){
    dataFile = std::move(dataPath);
    radio = _radio;
    k = _k;
    num_data_points = _num_data_points;
    num_search = _num_search;
    outputFile = std::move(outputPath);
    fromUser = _fromUser;
    myInput = _myInput;

    if(!is_open(dataFile, outputFile)) {
        cerr << "Error al abrir algún path mandado\n";
        exit(EXIT_FAILURE);
    }
}

void BaseModel::InitializeData(){


    auto start_b = std::chrono::steady_clock::now();

    // Open DataSet
    ifstream file;
    file.open(dataFile);

    // Cantidad total de datos en el DataSet.
    int TotalData = 0;
    string line;

    DataPoints.reserve(num_data_points);
    QueryPoints.reserve(num_search);

    if(fromUser){
        EDA::Point pt;
        pt.pt = trans->transformar(pt.pt);
        for(int idx = 0; idx < myInput.size(); ++idx)
            pt.set_point(idx, myInput[idx]);
        QueryPoints.push_back(pt);
        this->num_search = 1;
    }

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
        pt.pt = trans->transformar(pt.pt);

        for(int idx = 0; idx < cords.size(); ++idx)
            pt.set_point(idx, cords[idx]);


        if(TotalData < num_data_points)
            DataPoints.push_back(pt);
        else
            if(!fromUser) QueryPoints.push_back(pt);


        TotalData+=1;
        // if(fromUser && TotalData >= num_data_points) break;


    }

    file.close();
    cout << "Total de data points: " << TotalData << endl;
    cout << "Total de query points: " << num_data_points << endl;
    cout << "Total de puntos a buscar: " << num_search << endl;
    if(!fromUser && TotalData < num_data_points + num_search)
        throw runtime_error("No hay suficientes data para la busqueda.");
    if(fromUser && TotalData < num_data_points)
        throw runtime_error("No hay suficientes data para la busqueda2.");
    if(num_data_points < k)
        throw runtime_error("No hay suficientes data points para vecinos");

    cout << "Query Points: " << endl;
    for (const auto& query : QueryPoints) {
        cout << "- " << query << endl;
    }

    // Inicializamos los k vecinos.
    for(size_t idx = 0; idx < k*num_search; idx+=1){
        EDA::Neigh vecino{};
        vecino.dist = FLOAT_MIN;
        vecino.idx = -1;

        Vecinos.push_back(vecino);
    }

    auto end_b = std::chrono::steady_clock::now();
    auto elapsed_b = std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);

    TimeToInitializeData = elapsed_b.count()/1000000.0;

}
float BaseModel::get_radius() const {
    return radio;
}
int BaseModel::get_k() const {
    return k;
}
int BaseModel::get_num_search() const {
    return num_search;
}
void BaseModel::create_tree(CreateBVH* _gpu_process) {
    gpu_process = _gpu_process;
}
void BaseModel::obtain_results() {
    results = gpu_process->get_results();
}
void BaseModel::printf(std::ostream& os) {
    os << " \n\n\t ==================== Parametros KNN ==================== \n\n";
    os << "Radio: " << radio << "\n";
    os << "Numero de puntos en total a utilizar: "<< num_data_points << "\n";
    os << "Numero de query points a utilizar: " << num_search << "\n";
    os << "Numero de vecinos por query point: " << k << "\n";

    if(gpu_process->isTrueKNN())
        os << "Numero de rondas: " << gpu_process->GetNumRounds() << "\n";

    long long intersections = 0;
    for (int j = 0; j < num_search; j++)
        intersections += results[j * k].idx;


    os << " \n\n\t ==================== Tiempos ==================== \n\n";

    os << "Tiempo que tardo en procesar los datos: " << TimeToInitializeData << " segundos" << "\n";
    os << "Construccion del BVH: " << gpu_process->GetTimeCreateBVH() << " segundos"<< "\n";
    os << "Filter & Refine: " << gpu_process->GetTimeRayGen() << " segundos";

    os << " \n\n\t ==================== Guardando Resultados ==================== \n\n";

    os << "Numero de intersecciones: " << intersections << "\n";
    os << "Busqueda completada, escribiendo los resultados en el outputFile...\n";

    std::ofstream outfile(outputFile);

    // Cada k número de vecinos escritos serán los vecinos para el query point correspondiente
    // Los query points que son interesan son los que están en el rango [0:num_search] del total.
    os << "\nTotal de valores a escribir: " << num_search * k << "\n";
    outfile<<setprecision(4);

    for (int j = 0; j < num_search; j++) {
        for (int i = 0; i < k; i++) {

            // Cada grupo de k lines del OutputFile le pertenecerá a un Query Point.
            outfile << results[j * k + i].idx << '\t' << results[j * k + i].dist << '\n';
        }
    }

    os << "Escritura completa!";
    outfile.close();
}
void BaseModel::set_radius(float _radio) {
    radio = _radio;
}