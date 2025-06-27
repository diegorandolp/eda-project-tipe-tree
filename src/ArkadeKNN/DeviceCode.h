#ifndef OPTIX_OWL_DEVICECODE_H
#define OPTIX_OWL_DEVICECODE_H

#include <owl/owl.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>
#include <limits>
#include <iostream>

using namespace owl;
using namespace std;


// En esta sección se crearán las estructuras que se usaran en el proyecto.
// Serán las estructuras base para construir objetos personalizados de OWL.
namespace EDA{

    // Clase que guarda la información necesaria para el cálculo de los K vecinos más cercanos.
    struct Neigh{
        int idx;
        float dist;
    };

    // Punto en el espacio 3D
    struct Point{
        vec3f pt;

        void set_point(int idx, float val){

            if(idx > 3) {
                cerr << "El modelo Arkade no soporta más de 3 dimensiones.";
                exit(0);
            }

            pt[idx] = val;

        }
    };

    // Conjunto de esferas a utilizar.
    struct SpheresGeom{

        // Point se comporará como el circunscentro.
        Point* data_spheres;

        // Se especifica el radio de cada geometría (todas poseen el mismo radio)
        float rad;
    };

    // Se encarga de agrupar todos los query points y asignarle el mismo escenario de acción en el trazado de rayo.
    struct RayGenData{

        // Es un identificador para el rayo, especifica a que espacio de geometrías (BVH organizado) va a recorrer el rayo.
        OptixTraversableHandle world;

        // Arreglo con todos los Query Points.
        Point* query_pts;
    };

    // Se encargará de guardar variables globales en la GPU.
    struct GlobalVars{
        Neigh* frameBuffer;
        int k;
        int NORM;
    };

    // Clase que guardará los K vecinos más cercanos.
    struct NeighKNN{
        Neigh res[KN];
    };

}

#endif //OPTIX_OWL_DEVICECODE_H
