#include "FastRNN.h"

#include <utility>

float IncrementRadius(float radio, const string& distance){
    int d = 3;
    int p = stoi(distance);

    float exponent;
    if(p != 0) exponent = (1.0f / 2.0f) - (1.0f / static_cast<float>(p));

    // Para ceviche se saca raiz al # de dims.
    else exponent = 0.5f;

    return fmax(radio, radio * pow(static_cast<float>(d), exponent));
}
FastRNN::FastRNN(std::string dataPath, string _distance, float _radio, int _k, int _num_data_points, int _num_search,
                 bool TrueKnn,
                 std::string outputPath, bool fromUser, vector<float> myInput) {

    // TrueKNN se basa en la norma L2
    distance = std::move(_distance);

    if (!esEntero(distance)) {
        cerr << "FastRNN solo soporta distancias L^p con p entero positivo." << endl;
        exit(EXIT_FAILURE);
    }

    // Aumentar r'
    float ExpandRadio = _radio;
    ExpandRadio = IncrementRadius(_radio, distance);

    // Aumentar k'
    float factor_k = 1.f + 0.5 * sqrt(3);
    _k = ceil(_k * factor_k);

    // Esto se debe a que FastRNN trabaja únicamente con la distancia euclidiana.
    distance = "Euclidian";
    trans = new TransMonotoma(distance);
    if(fromUser) {
        baseModel = new BaseModel(std::move(dataPath), ExpandRadio, _k, _num_data_points, 1, std::move(outputPath), fromUser, myInput);
    } else {
        baseModel = new BaseModel(std::move(dataPath), ExpandRadio, _k, _num_data_points, _num_search, std::move(outputPath), fromUser, myInput);
    }
    bool isTrueKnnModel = TrueKnn;
    auto* gpu_process = new CreateBVH(baseModel, isTrueKnnModel, distance);
    baseModel->create_tree(gpu_process);
    baseModel->obtain_results();

    // to print the results
    cout << *this << endl;
}
