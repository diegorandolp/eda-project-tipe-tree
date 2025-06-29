// All rights reserved by
// Durga Keerthi Mandarapu, Vani Nagarajan, Artem Pelenitsyn, and Milind Kulkarni. 2024.
// Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing.

#include "deviceCode.h"
#include <optix_device.h>

using namespace owl;


// Acá agarramos la variable global owl creada en el MAIN.
__constant__ MyGlobals optixLaunchParams;

// bounding box programs
template<typename SphereGeomType>

// lo que esté con __device__ se ejecuta en la GPU

inline __device__ void boundsProg(const void *geomData,
                                  box3f &primBounds,
                                  const int primID)
{
  const SphereGeomType &self = *(const SphereGeomType*)geomData;

  // rescatamos la esfera exacta
  const Sphere sphere = self.data_pts[primID];

    // Debemos crear el AABB apartir de la geometría para que ocupe exactamente el espacio que ocupa
    // box3f() crea una caja vacia, extend extiende CADA EJE DE LA ESFERA en - rad,
    // extenel 2do extend, extiende cada eje de la esfera ahora en + rad, cubriendo exactamente la esfera

    // tal que al excetender el box3f, está cubriendo exactamente la geometría.
  primBounds = box3f().extend(sphere.center - self.rad)
		                  .extend(sphere.center + self.rad);

}

// Si es que quiero usar una geometría personalizada, debo crear mi macro para que el AABB del BVH, sepa como englobal las geometrías
// acá está creando el AABB del esferas.
OPTIX_BOUNDS_PROGRAM(Spheres)(const void  *geomData,
  box3f       &primBounds,
  const int    primID)

    //geomData es spheresGroup, internamente se guarda como void, si se quiere acceder asus valores:
    //   const SphereGeomType &self = *(const SphereGeomType*)geomData;

    // primID: es el indice de la esfera la cual se está procesando y entrando a la función.
{ boundsProg<SpheresGeom>(geomData,primBounds,primID); }


// Este macro es el proceso de intersección de una geometría y un AABB.
// Corazón del recorrido e intersección personalizada con el BVH.

// Cuando llegas acá, es todo el proceso de RT Cores
OPTIX_INTERSECT_PROGRAM(Spheres)()
{

    // El optixGetPrimitiveIndex() devuelve el rayo que está intersectando en ese momento
    // getProgramData() devuelve el conjunto de datos principal del programa, en este caso el conjunto de esferas
	const int primID = optixGetPrimitiveIndex();
	const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();

    // extraigo la esfera ctual (Shader Core)
	Sphere self = selfs.data_pts[primID];

    // Te deuelve el punto donde se lanzó el rayo, osea la query
	const vec3f org = optixGetWorldRayOrigin();

    // Todo lo de arriba el Shader Core, sacamos los atributos que necesitamos, los que ocnseguimos con ayuda del BVH
    // ahora solo calcularemos los mejores vecinos.

	float distance = 0.0;

    // Acá definimos la norma a usar.
//#define NORM 0
//#define KN 20


    // Función de distancia que se va a usar en la NORM L^p
    // entre el centro de la geometría y la query.

#if (NORM == 0) // ceviche
	double x  = std::abs(self.center.x - org.x);
	double y = std::abs(self.center.y - org.y);
	double z = abs(self.center.z - org.z);
	if(x > y )
		distance = x;
	else
		distance = y;
	if(distance < z)
		distance = z;
#elif (NORM > 0) // p >= 1 pero no ceviche

    // no sacamos raiz, AÚN
  	distance = std::pow(std::abs(self.center.x - org.x), NORM)
			 + std::pow(std::abs(self.center.y - org.y), NORM)
			 + std::pow(std::abs(self.center.z - org.z), NORM);
#endif

      // Esta es la parte de filtrado, todos los que entren acá son posibles candidatos.
      // Acá se eleva al cuadrado porque estamos en el supuesto de que estamos usando norma euclidiana.
      // sería lo equivalente a poner srqt(distance, NORM) < self.rad
	if(distance < pow(selfs.rad, NORM)){

        NeighGroup &param = owl::getPRD<NeighGroup>();

        // Acá simplemente rescatamos el peor vecino, para comparar con el nuevo vecino actual
        // y si ver si es mejor que el peor vecino registrado.
        int max_idx=0;
        for (int i = 1; i < KN; i++){
          if (param.res[i].dist > param.res[max_idx].dist)
            max_idx = i;
        }

        // Parte de refinamiento, si cumple, refinamos los vecinos que ya teniamos.
        if ( distance < param.res[max_idx].dist) {

            // En el max_idx, guarda el peor vecino, acá guardamos el vecino actual, que es mejor que el peor.
            // lo guardamos tanto el indice DEL BLOQUE como tambien su distancia.
          param.res[max_idx].dist = distance;
          param.res[max_idx].ind = primID;
        }
  }

}


// Esto se activa internamente cuando se llama a owlLaunch2D
OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    // Rescatamos el rayo de la query
  const RayGenData &self = owl::getProgramData<RayGenData>();

  // Esto es fundamental para definir las mallas, en la lanzada de rayos en dif dimensiones.
  // SERÍA LOS puntos de consulta, lo que va a procesar el GPU en el lanzado de rayos.
  int xID = optixGetLaunchIndex().x;

  // Inicializo los vecinos con un valor incial.
  NeighGroup param;
  for(int i=0; i<KN; i++){
    param.res[i].ind = -1;
    param.res[i].dist = FLOAT_MAX;
  }

  // configuro el rayo a lanzar, con los datos de la query
  owl::Ray ray(self.query_pts[xID].center, vec3f(0,0,1), 0, 1.e-16f);

  // trazo el rayo en el escenario creado, guardando los vecinos en param.
  // esto llama a OPTIX_INTERSECT_PROGRAM, e
  owl::traceRay(self.world, ray, param);

  for(int i=0; i<KN; i++){
      // Esto guarda los resultados en la variable global, básicamente hace lo siguiente:
      // en base al punto de procesamiento xID, pues multiplixa con Kn + i, para decir
      //  xID = 0 [0, 19] xID = 1 [20, 39] xID= 2 [40, 59] ...
    optixLaunchParams.frameBuffer[xID*KN+i].ind = param.res[i].ind;
    optixLaunchParams.frameBuffer[xID*KN+i].dist = param.res[i].dist;
  }
}
