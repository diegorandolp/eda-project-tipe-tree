//
// Created by RODRIGO on 28/06/2025.
//

#ifndef OPTIX_OWL_BASEDMODEL_H
#define OPTIX_OWL_BASEDMODEL_H


#include "CreateBVH.h"

#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <set>
#include <iomanip>
#include <ios>
#include "VarGlobal.h"

class CreateBVH;

struct BaseModel{
private:
    string dataFile;
    float radio{};
    int k{};
    int num_data_points{};
    int num_search{};
    string outputFile;

    // Tiempo que tardó en inicializar los datos
    float TimeToInitializeData{};

    // Instancia a la clase CreateBVH
    CreateBVH* gpu_process;

    // Resultados
    const EDA::Neigh* results;

public:
    BaseModel(string dataPath, float _radio, int _k, int _num_data_points,
        int _num_search, string outputPath);

    void InitializeData();
    float get_radius() const;
    int get_k() const;
    int get_num_search() const;
    void create_tree(CreateBVH* _gpu_process);
    void obtain_results();
    void printf(std::ostream& os);
    void set_radius(float _radio);

};


#endif //OPTIX_OWL_BASEDMODEL_H
