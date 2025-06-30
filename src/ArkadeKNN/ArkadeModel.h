//
// Created by RODRIGO on 20/06/2025.
//

#ifndef OPTIX_OWL_ARKADEMODEL_H
#define OPTIX_OWL_ARKADEMODEL_H

#include "BaseModel.h"

class ArkadeModel {
private:

    // Instancia a la clase padre.
    BaseModel* baseModel;

    // Nombre de la distancia.
    string distance{};


public:


    ArkadeModel(string dataPath, string _distance, float _radio, int _k, int _num_data_points,
                int _num_search, bool TrueKnn, string outputFile);

    friend std::ostream& operator<<(std::ostream& os, const ArkadeModel& model){
        cout << " \n\n\t ==================== Distancia A Utilizar ==================== \n\n";
        os << model.distance;
        model.baseModel->printf(os);
        return os;
    }

};


#endif //OPTIX_OWL_ARKADEMODEL_H
