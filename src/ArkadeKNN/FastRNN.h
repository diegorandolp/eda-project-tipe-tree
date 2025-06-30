//
// Created by RODRIGO on 28/06/2025.
//

#ifndef OPTIX_OWL_FASTRNN_H
#define OPTIX_OWL_FASTRNN_H

#include "BaseModel.h"

// Con TrueKNN
class FastRNN {
private:
    BaseModel* baseModel;
    string distance;
public:
    FastRNN(string dataPath, string distance, float _radio, int _k, int _num_data_points,
            int _num_search, bool TrueKnn, string outputPath, bool fromUser, vector<float> myInput);

    friend std::ostream& operator<<(std::ostream& os, const FastRNN& model){
        cout << " \n\n\t ==================== Distancia A Utilizar ==================== \n\n";
        os << "Euclidiana";
        model.baseModel->printf(os);
        return os;
    }
};


#endif //OPTIX_OWL_FASTRNN_H
