//
// Created by RODRIGO on 20/06/2025.
//

#ifndef OPTIX_OWL_ARKADEMODEL_H
#define OPTIX_OWL_ARKADEMODEL_H


#include "VarGlobal.h"
#include "CreateBVH.h"
#include "TransMonotoma.h"

#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <set>
#include <iomanip>
#include <ios>

class CreateBVH;

class ArkadeModel {
private:

    // Parámetros KNN
    string dataFile;
    float radio{};
    int k{};
    int num_data_points{};
    int num_search{};
    string outputFile;

    // Nombre de la distancia.
    string distance{};

    // transformación monótoma
    TransMonotoma* func;

    // Instancia a la clase CreateBVH
    CreateBVH* gpu_process;

    // Resultados
    const EDA::Neigh* results;

public:


    ArkadeModel(string dataPath, string _distance, float _radio, int _k, int _num_data_points,
                int _num_search, string outputFile);

    void InitializeData();

    float get_radius() const;

    int get_k() const;

    int get_num_search() const;

    string& get_distance_type();

    friend std::ostream& operator<<(std::ostream& os, const ArkadeModel& model){

        cout << " \n\n\t ==================== Parametros KNN ==================== \n\n";
        std::cout << "Radio: " << model.radio << "\n";
        std::cout << "Numero de puntos en total a utilizar: "<< model.num_data_points << "\n";
        std::cout << "Numero de query points a utilizar: " << model.num_search << "\n";
        std::cout << "Numero de vecinos por query point: " << model.k << "\n";
        std::cout << "Distancia utilizada: " << model.distance;

        long long intersections = 0;
        for (int j = 0; j < model.num_search; j++)
            intersections += model.results[j * model.k].idx;


        cout << " \n\n\t ==================== Tiempos ==================== \n\n";

        cout << "Construccion del BVH: " << model.gpu_process->GetTimeCreateBVH() << " segundos"<< "\n";
        cout << "Filter & Refine: " << model.gpu_process->GetTimeRayGen() << " segundos";


        cout << " \n\n\t ==================== Guardando Resultados ==================== \n\n";

        os << "Numero de intersecciones: " << intersections << "\n";
        os << "Busqueda completada, escribiendo los resultados en el outputFile...\n";

        std::ofstream outfile(model.outputFile);

        // Cada k número de vecinos escritos serán los vecinos para el query point correspondiente
        // Los query points que son interesan son los que están en el rango [0:num_search] del total.
        os << "\nTotal de valores a escribir: " << model.num_search * model.k << "\n";

        for (int j = 0; j < model.num_search; j++) {
            for (int i = 0; i < model.k; i++) {

                // Cada grupo de k lines del OutputFile le pertenecerá a un Query Point.
                outfile << model.results[j * model.k + i].idx << '\t' << model.results[j * model.k + i].dist << '\n';
            }
        }

        cout << "Escritura completa!";
        outfile.close();

        return os;
    }

};


#endif //OPTIX_OWL_ARKADEMODEL_H
