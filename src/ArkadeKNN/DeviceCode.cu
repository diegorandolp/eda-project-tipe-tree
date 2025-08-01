// All rights reserved by
// Durga Keerthi Mandarapu, Vani Nagarajan, Artem Pelenitsyn, and Milind Kulkarni. 2024.
// Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing.

#include "DeviceCode.h"
#include "VarGlobal.h"
#include <optix_device.h>
#include <cuda_runtime.h>

using namespace owl;



// Variable global constante accedida desde el main
__constant__ EDA::GlobalVars optixLaunchParams;

__device__ bool FueVecinoAntes(int xID, int primID){

    int k = optixLaunchParams.k;
    for(int i = 0; i < k; ++i){
        int SearchVecino = optixLaunchParams.frameBuffer[xID * k + i].idx;

        if(SearchVecino == primID) return true;
    }

    return false;
}

// Template para soportar múltiples tipos de geometrías de esfera
template<typename SphereGeomType>

// ============================================================================
// Bounds Program - Se ejecuta en la GPU y calcula el AABB (Axis-Aligned Bounding Box)
// para una primitiva de tipo esfera.
// ============================================================================
inline __device__ void boundsProg(const void* geomData,
                                  box3f& primBounds,
                                  const int primID){
    // Cast del puntero genérico a nuestra estructura de geometría
    const SphereGeomType& self = *(const SphereGeomType*)geomData;

    // Obtenemos la esfera específica usando el ID de la primitiva
    const EDA::Point sphere = self.data_spheres[primID];

    // Creamos la bounding box (AABB) que encapsula exactamente la esfera.
    // Se extiende el box vacío en ambas direcciones a partir del centro:
    //  - primero hacia (centro - radio)
    //  - luego hacia (centro + radio)
    //
    // Esto genera una caja perfectamente ajustada a la esfera.
    primBounds = box3f()
            .extend(sphere.pt - self.rad)
            .extend(sphere.pt + self.rad);
}

// ============================================================================
// Crea el AABB exacto para cada esfera usando el programa de bounds de OptiX.
// Este macro enlaza tu función personalizada `boundsProg` con el sistema de BVH.
// ============================================================================
OPTIX_BOUNDS_PROGRAM(Spheres)(const void  *geomData, box3f &primBounds, const int primID){

    // geomData apunta al grupo de esferas (es un void* internamente)
    // primID: indice para acceder a la geometría que se está procesando en ese momento.
    boundsProg<EDA::SpheresGeom>(geomData,primBounds,primID);
}


// ============================================================================
// Intersección personalizada entre un rayo y una esfera.
// Este código se ejecuta cuando un rayo entra en una AABB y necesita decidir
// si realmente intersecta la geometría (esfera).
// En este MACRO se ejecuta el algoritmo de Filter & Refine para un Query Point.
// Acá se activa los RT Cores para la intersección de un Query RayGen y un AABB
// ============================================================================
OPTIX_INTERSECT_PROGRAM(Spheres)(){

    // ID de la primitiva con la que se está intentando intersectar en ese momento
    const int primID = optixGetPrimitiveIndex();

    // Obtenemos el conjunto de datos de la geometría actual
    const auto &self = owl::getProgramData<EDA::SpheresGeom>();

    // Extraemos la esfera correspondiente (ShaderCore)
    const EDA::Point sphere = self.data_spheres[primID];

    // Punto de origen del rayo en coordenadas del mundo (ShaderCore)
    const vec3f rayOrigin = optixGetWorldRayOrigin();

    // Extraer la norma a utilizar de las variables globales.
    int NORM = optixLaunchParams.NORM;

    bool isTrueKnn = optixLaunchParams.isTrueKnn;

    int k = optixLaunchParams.k;

    // Accedemos al registro de vecinos actual
    auto& param = owl::getPRD<EDA::NeighKNN>();
    const int xID = optixGetLaunchIndex().x;

    // En caso de estar usando el modelo TrueKnn, se realiza una verificación adicional:
    // - Si estamos en la ronda 0, simplemente se asignan los vecinos iniciales a cada query (sin filtrado).
    // - Si estamos en una ronda > 0, se verifica si este data point ya fue vecino en la ronda anterior.
    //   Si ya fue mapeado previamente como vecino, se descarta, ya que no es necesario refinarlo nuevamente.
    if(isTrueKnn && optixLaunchParams.round > 0 && FueVecinoAntes(xID, primID))
        return;

    // -----------------------------
    // Cálculo de distancia (L^p Norm)
    // -----------------------------
    float distance = 0.0f;

    if (NORM == 0) { // Norm infinito (máxima componente absoluta)
        float dx = std::abs(sphere.pt.x - rayOrigin.x);
        float dy = std::abs(sphere.pt.y - rayOrigin.y);
        float dz = std::abs(sphere.pt.z - rayOrigin.z);
        distance = fmaxf(fmaxf(dx, dy), dz);
    }
    else if (NORM > 0) {
        // Norma L^p (sin raíz para eficiencia)
        distance = powf(std::abs(sphere.pt.x - rayOrigin.x), NORM) +
                   powf(std::abs(sphere.pt.y - rayOrigin.y), NORM) +
                   powf(std::abs(sphere.pt.z - rayOrigin.z), NORM);
    }

    // -----------------------------
    // Filtro de candidatos por radio (Filter)
    // Cálculo equivalente a sqrt(distance, NORM) < self.rad
    // -----------------------------

    if (!isTrueKnn) {
        if ((NORM > 0 && distance > powf(self.rad, NORM)) ||
            (NORM == 0 && distance > self.rad))
            return;
    }

    // Buscamos el vecino más lejano en la lista actual, el peor registrado.

    int max_idx = 0;
    for (int i = 1; i < k; ++i) {
        if (param.res[i].dist > param.res[max_idx].dist)
            max_idx = i;

    }

    // Si la nueva distancia es mejor que el peor vecino, lo reemplazamos (Refine)
    if (distance < param.res[max_idx].dist) {
        param.res[max_idx].dist = distance;
        param.res[max_idx].idx  = primID;
        param.res[max_idx].original_idx = sphere.original_idx;
    }

}


// ============================================================================
// Programa de generación de rayos (rayGen).
// Se activa automáticamente en cada índice de lanzado al ejecutar owlLaunch2D().
// Este lanzador recorre los puntos de consulta (query points) y realiza trazado de rayos.
// ============================================================================
OPTIX_RAYGEN_PROGRAM(rayGen)(){

    // -----------------------------
    // 1. Obtener datos del programa
    // -----------------------------
    const auto& self = owl::getProgramData<EDA::RayGenData>();

    // Índice X de la RayGen Query lanzado (cada hilo procesa un punto de consulta)
    const int xID = optixGetLaunchIndex().x;

    // Para el TrueKnn, si ya se mapeo en esa query K vecinos, no tiene sentido continuar con el proceso.
    // solo es para los que tengan < KN vecinos.


    int k = optixLaunchParams.k;
    if(optixLaunchParams.num_neighbors[xID] >= k) return;

    // -----------------------------
    // 2. Inicializar los vecinos (KNN)
    // -----------------------------
    EDA::NeighKNN param;

    for (int i = 0; i < k; ++i) {
        param.res[i].idx = -1;
        param.res[i].original_idx = -1;
        param.res[i].dist = FLOAT_MAX;
    }


    // -----------------------------
    // 3. Construcción del rayo
    // -----------------------------
    // Se lanza un rayo desde el punto de consulta en dirección arbitraria (por ejemplo +Z)
    owl::Ray ray(self.query_pts[xID].pt, // Origen del rayo
                 vec3f(0,0,1), // Dirección (arbitraria en este contexto)
                 0, // Tipo del rayo
                 1.e-16f); // Para evitar auto intersecciones.

    // -----------------------------
    // 4. Trazado del rayo (intersección con la escena)
    // -----------------------------
    // Esto inicia el recorrido por el BVH y llama a OPTIX_INTERSECT_PROGRAM
    // param es modificado dentro del programa de intersección con los KNN encontrados

    owl::traceRay(self.world, ray, param);

    // -----------------------------
    // 5. Guardar resultados en el framebuffer global
    // En el Buffer de los resultados tiene num_search * KN de espacio reservado, para guardar por indices
    // los vecinos más cercanos de CADA query point.
    // -----------------------------

    // Para el TrueKnn

    int num_neighbors = 0;
    for (int i = 0; i < k; i+=1) {


        // Para el TrueKnn, para el modelo Arkade nunca la incumplirá (para verificar posibles falsos positivos)
        if(param.res[i].idx != -1) {
            // Cada k vecinos conseguidos le pertenece al query point correspondiente
            // El rango de índices en el que se ubica sus vecinos es: [ID*KN : ID*(KN+1)]
            int outputIndex = xID * k + i;
            optixLaunchParams.frameBuffer[outputIndex].idx = param.res[i].idx;
            optixLaunchParams.frameBuffer[outputIndex].original_idx = param.res[i].original_idx;
            optixLaunchParams.frameBuffer[outputIndex].dist = param.res[i].dist;

            num_neighbors += 1;
        }

    }

    //printf("num_vecinos: %d\n", num_neighbors);
    //printf("vecinos: %d\n", k);

    // Seteamos el # de vecinos detectados para la query xID.

    optixLaunchParams.num_neighbors[xID] = num_neighbors;
    //printf("vec: %d\n", optixLaunchParams.num_neighbors[xID]);

}
