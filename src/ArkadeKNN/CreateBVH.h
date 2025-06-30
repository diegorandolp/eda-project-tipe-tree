//
// Created by RODRIGO on 20/06/2025.
//

#ifndef OPTIX_OWL_CREATEBVH_H
#define OPTIX_OWL_CREATEBVH_H

#include <owl/owl.h>
#include <owl/DeviceMemory.h>
#include "VarGlobal.h"

class BaseModel;

class CreateBVH {
private:
    BaseModel* model;

    // Parámetros OWL (para la creación del BVH)
    OWLContext context;
    OWLModule  module;
    const EDA::Neigh* results;
    string distance{};
    int round;

    // Medido en segundos.
    float TimeCreateBVH{};
    float TimeRayGen{};

    bool isTrueKnnModel;

    vector<int> n_neighbors;


public:
    explicit CreateBVH(BaseModel* _model, bool _isTrueKnnModel, string _distance = "2");

    const EDA::Neigh* GetResultsAndCreateBVH();

    ~CreateBVH();

    const EDA::Neigh* get_results();

    float GetTimeCreateBVH();
    float GetTimeRayGen();
    int GetNumRounds();
    bool isTrueKNN();

};


#endif //OPTIX_OWL_CREATEBVH_H
