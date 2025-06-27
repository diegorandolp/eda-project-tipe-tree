//
// Created by RODRIGO on 20/06/2025.
//

#ifndef OPTIX_OWL_CREATEBVH_H
#define OPTIX_OWL_CREATEBVH_H

#include <owl/owl.h>
#include <owl/DeviceMemory.h>
#include "VarGlobal.h"

class ArkadeModel;

class CreateBVH {
private:
    ArkadeModel* model;

    // Parámetros OWL (para la creación del BVH)
    OWLContext context;
    OWLModule  module;
    const EDA::Neigh* results;

    // Medido en segundos.
    float TimeCreateBVH{};
    float TimeRayGen{};


public:
    explicit CreateBVH(ArkadeModel* _model);

    const EDA::Neigh* GetResultsAndCreateBVH();

    ~CreateBVH();

    const EDA::Neigh* get_results();

    float GetTimeCreateBVH();
    float GetTimeRayGen();

};


#endif //OPTIX_OWL_CREATEBVH_H
