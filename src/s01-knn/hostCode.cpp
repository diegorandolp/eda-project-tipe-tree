// All rights reserved by 
// Durga Keerthi Mandarapu, Vani Nagarajan, Artem Pelenitsyn, and Milind Kulkarni. 2024. 
// Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing. 

// public owl API
#include <owl/owl.h>
#include <owl/DeviceMemory.h>
#include "deviceCode.h"          

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <set>
#include <iomanip>
#include <ios>

#define FLOAT_MAX 3.402823466e+38
#define FLOAT_MIN 1.175494351e-38

extern "C" char deviceCode_ptx[];

std::vector<Sphere> Spheres_data;
std::vector<Sphere> Spheres_query;
std::vector<Neigh> neighbors;

//#define KN 50

int main(int ac, char **argv)
{
  std::string line;
  std::ifstream myfile;

  myfile.open(argv[1]);
  int npoints = atoi(argv[2]);
  int nsearchpoints = atoi(argv[3]);
  float radius = atof(argv[4]); 
  int knn = 20;

  if(!myfile.is_open())
  {
    perror("Error open");
    exit(EXIT_FAILURE);
  }
  int linenum = 0;

  // Acá extrae la información del txt
  while(getline(myfile, line)) 
	{
	  std::stringstream ss(line);
	  float i;
    std::vector<float> vect;

    // Se espera que sea de 2d o 3d
	  while (ss >> i)
	  {
      vect.push_back(i);
      if (ss.peek() == ',')
          ss.ignore();
	  }

      // Si es que se tienen 5k de datos y se ponen 3k como puntos, pues:
      // los primeros 3k como data points y los otros 2k como query points.
    if(linenum < npoints)
      // Spheres_data.push_back(Sphere{vec3f(vect[0],vect[1],vect[2])});
      Spheres_data.push_back(Sphere{vec3f(vect[0],vect[1],0.0f)});
    else
      // Spheres_query.push_back(Sphere{vec3f(vect[0],vect[1],vect[2])});
      Spheres_query.push_back(Sphere{vec3f(vect[0],vect[1],0.0f)});
    
    linenum++;
	}	
  myfile.close();


  // Si la cantidad de buscar, son mayor que la totalidad de los puntos, pues ERROR.
  if(linenum < (npoints+nsearchpoints)){
    printf("Insufficient file size\n");
    return 0;
  }

	// Inicializa todos los vecinos con la longitud mínima.
	for(int j=0; j<nsearchpoints; j++){
    for(int i = 0; i < knn; i++)
		  neighbors.push_back(Neigh{-1,FLOAT_MIN});
  }

  std::cout << "Radio: " << radius << std::endl;
  std::cout << "K vecinos: " << knn << std::endl;
  std::cout << "Numero de puntos utilizados: "<< npoints << std::endl;
  std::cout << "N busquedas por punto:" << nsearchpoints << std::endl << std::endl;

	//Frame Buffer -- just one thread for a single query point
	const int fbSize = nsearchpoints;


  // Primer parámetro: Crea el contexto en el cual optix lanzará el rayos
  // Segundo parámetro: el número de GPU's que usará ( 1 en este caso )
  OWLContext context = owlContextCreate(nullptr,1);

  // Modulo que me permite manejar los aspectos de CUDA de forma personalizada, en base al contexto del lanzado de rayos
  // el devideCode <- carga el código cuda compilado.
  OWLModule  module  = owlModuleCreate(context, deviceCode_ptx);

  // Declarando las variables en OWL para las geometrías.
  // Esa clase creada por mi, se usará como base para crear una geometría basada en optix
  // donde el buffer se irá en data_pts, y el radio en rad, para su uso con el GPU
  OWLVarDecl SpheresGeomVars[] = {
    { "data_pts",  OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom, data_pts)},
    { "rad", OWL_FLOAT, OWL_OFFSETOF(SpheresGeom,rad)},
    { /* sentinel to mark end of list */ }
  };

  // Apartir de esto, creo como una instancia para poder usarla conjuntamente con la arquitectura de optix
  // especificando el comportamiento de geometría, el tamaño en bytes de la clase personalizada como el arreglo de clases
  // con un -1 para decir que no hay limite de rayos simultaneos

  // Es como pytorch, si es que se quiere crear una clase personalizada que se pueda usar en conjunto con sus librerias de pytorch
  // se debe hacer x cosas, acá estamos logrando el mismo fin pero con Optix
  OWLGeomType SpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(SpheresGeom),
                        SpheresGeomVars,-1);


  // Permite personalizar las intersecciones a esferas en el modulo creado, para las intersecciones de rayos a geometrías
  owlGeomTypeSetIntersectProg(SpheresGeomType,0,
                              module,"Spheres");

  // Sirve para especificar cual será la geometría que el AABB va a envolver en el arbol BVH, se puede manejar por separado
  // o directo con la linea anterior, en el modulo especificado
  owlGeomTypeSetBoundsProg(SpheresGeomType,
                           module,"Spheres");

  // estas 2 ultimas internamente manejan que geometría deben envolver los AABB, como el tipo de intersección que tendrá un rayo
  // internamente manejan las geometrías mediante las clases personalizadas configuradas con optix, para la construcción del arbol
  // BVH


  // No crea el arbol, si no, en base a la configuración del modulo y contexto previamente, agarra esto y configura la clasee
  // del BVH para crearlo apartir de este formato especficado.
  owlBuildPrograms(context);
  // COn esto, si es que quiero cambiar a otra geometría, que ya no sea esfera, si no otra, pues solamente sería
  // cambiar el Sphere a otra geometría permitida por Optix como cambiar la clase personalizada (struct).


  // El buffer permite la comuniación entre la CPU y GPU de forma eficiente, traslada datos, en este caso (en la query y data)
  // de la CPU a la GPU para poder usarse en el contexto.

  // Crea un espacio de memoria en la GPU. <- esto se encargará de describir los resultados del KNN
  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_USER_TYPE(neighbors[0]),
                            neighbors.size());

  // En base a la configuración de la data, se extrae las caracteristcias para poder llegar a guardar
  // en la GPU todo el contenido relacionado con la data, además de que se pueda llegar a acceder y usar en el contexto
  // declarado.
  OWLBuffer dataSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(Spheres_data[0]),
                            Spheres_data.size(),Spheres_data.data());

  OWLBuffer querySpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(Spheres_query[0]),
                            Spheres_query.size(),Spheres_query.data());

    // Gramática: owlHostPinnedBufferCreate(contexto - entorno, type owl, size) <- crea buffer ( memoria fijada <- pinned ) en la GPU
    // owlDeviceBufferCreate(contexto - entorno, type owl, size, data) transporta la data de la CPU a la GPU (dispositivo <- mio)



    // Crea una instancia de la geometría personalizada dentro del contexto de OWL.
    // Esta instancia representa una colección de primitivas (esferas) que será usada para construir la escena.
    OWLGeom SpheresGeom = owlGeomCreate(context,SpheresGeomType);

    // En este paso se inicializa la geometría en la GPU. Esta geometría funcionará como un contenedor de todas
    // las primitivas definidas en el buffer `dataSpheresBuffer`, que ya reside en la memoria de la GPU.
    // Se configura la cantidad de elementos que contendrá y el radio de búsqueda que usará cada esfera
    // durante la fase de intersección en el k-NN.

    // Asocia el buffer de datos `dataSpheresBuffer` (ya cargado en la GPU) al campo "data_pts" (shader core) de la geometría.
// Este campo será accedido desde los programas de intersección en el dispositivo.
  owlGeomSetPrimCount(SpheresGeom, Spheres_data.size());

  // Setea todas los datos de las data geometry que está en la GPU a la geometría.
  owlGeomSetBuffer(SpheresGeom,"data_pts",dataSpheresBuffer);

    // Define el valor del radio de búsqueda (`radius`) que será utilizado por el shader en cada intersección entre el rayo y
    // la geometría.
  owlGeomSet1f(SpheresGeom,"rad",radius);

  // Parametros globales del modelo a utilizar
  // Es para hacer una conexión entre CUDA y el c++ indicando las variables globales a usar
  // Se generan estos parámetros globales apartir deuna clase personalizada.

  // Sintaxis {name_param, owl type paran, OWL_OFFSETOF(class personalizada, atribute class)}
  OWLVarDecl myGlobalsVars[] = {
	{"frameBuffer", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, frameBuffer)},
	{"k", OWL_INT, OWL_OFFSETOF(MyGlobals, k)},
	{ /* sentinel to mark end of list */ }
	};

    // Crea el objeto que contendrá los parámetros globales, con todos los declarados
    // en el contexto especificado.
	OWLParams lp = owlParamsCreate(context,sizeof(MyGlobals),myGlobalsVars,-1);

    // Setea los valores de los parámetros en la variable global owl, con los valores esperados...
	owlParamsSetBuffer(lp,"frameBuffer",frameBuffer);	
	owlParamsSet1i(lp,"k",knn);		

    // Las geometrías que se configuraron en 1 sola geometría, se utilizará para crear el arbol BVH
  OWLGeom  userGeoms[] = {
    SpheresGeom
  };

	auto start_b = std::chrono::steady_clock::now();

    // Este es el grupo de geometrías personalizadas las cuales se van a usar y agrupar en el espacio
    // En este caso, como solo estamos usando geometrías básicas de esferas, ponemos solamente 1 grupo
    // se pondría acá el userGeoms.size()

    // Gramática --> owlUserGeomGroupCreate(context - entorno, num geometry groups, groups array)
  OWLGroup spheresGroup = owlUserGeomGroupCreate(context,1,userGeoms);

  // Con esto genera el BVH que poda el espacio de las geometrías especificadas por el usuario
  // PARA CREAR LAS GEOMETRÍAS PERSONALIZADAS, SE USA EL MACRO OPTIX_BOUNDS_PROGRAM
  owlGroupBuildAccel(spheresGroup);

  // Esto genera el mundo o escenario de todas las geometrías creadas, en este caso solo tengo 1 grupo
  // de geometrías, así que pongo solo 1, en el contexto

  // Esto solo crea el escenario de la instancia sphere group, así que podría modificar el mismo spheresGroup y
  // seguir utilizando esta instancia
  OWLGroup world = owlInstanceGroupCreate(context,1,&spheresGroup);

  // Construimos el arbolito BVH apartir de la instancia del grupo de geometrías.
  owlGroupBuildAccel(world);

  auto end_b = std::chrono::steady_clock::now();

  // Este tiempo se encarga de medir cuanto tiempo tarda en construir el arbol (teniendo todo configurado anteriormente)
	auto elapsed_b = std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
	std::cout << "Construcción del arbol BVH: "<< elapsed_b.count()/1000000.0 << std::endl;


    // Seteamos los parámetros que necesita el lanzado de rayos, mediante una clase personalizada.
    // Le pasamos los query sphere como tambien el escenario.
    // A cada parámetro sería como {name atribute, type owl, atribute class personalizada}
	OWLVarDecl rayGenVars[] = {
    {"query_pts",       OWL_BUFPTR, OWL_OFFSETOF(RayGenData,query_pts)},
		{ "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},	
		{ /* sentinel to mark end of list */ }
	};

    // Generamos los rayos, con los query points, el escenario, contexto y modulo.
    // Con esto nomás creamos el objeto owl con los atributos de nuestra clase personalizada.
    // Gramática -> owlRayGenCreate(context - escenario, module, type action, num bits de la clase personalizada, variable owl de rayos)
	OWLRayGen rayGen
		= owlRayGenCreate(context,module,"rayGen",
			                sizeof(RayGenData),
			                rayGenVars,-1);

  // ----------- set variables  ----------------------------
  // Acá seteamos las variables de la clase personalizada con owl
  owlRayGenSetBuffer(rayGen,"query_pts",        querySpheresBuffer);
	owlRayGenSetGroup (rayGen,"world",        world);
	
  // build shader binding table required to trace the groups

  // Construye todo el escenario, con todas las caracteristicas colocadas. Compula funciones de Optix.
  owlBuildPrograms(context);

  // Esto arma la estructura interna para que optix pueda realizar todo el proceso
	owlBuildPipeline(context);

    // A cada objeto diferente de todo el espacio o contexto, se le ancla su función en especifico para que la realice
    // en la gpu. FUNDAMENTAL. Rescata esas acciones y las ejecuta en la GPU del contexto cuando sea necesario.
  owlBuildSBT(context);
	
  auto start = std::chrono::steady_clock::now();

    // Se encarga de lanzar los rayos al programa.


    // Parámetros:
    // raygen: en base a OPTIX_RAYGEN_PROGRAM, se puede acceder al llamado de rayo y ejecutar la acción con CUDA.
    // nsearchpoints: Número de rayos que estoy lanzando en total, que sería igual al número de puntos de la query
    // un rayo por query point.
    // 1: ...
    // lp: variable global owl, donde se guardarán los resultados.

    // Cómo es el recorrido del arbol con la query (rayos), se activa el macro de OPTIX_INTERSECT_PROGRAM
  owlLaunch2D(rayGen,nsearchpoints,1,lp);

  auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Trazado de rayos de la Query: " << elapsed.count()/1000000.0<< std::endl;


    // Rescatamos los resultados del buffer, acá se almacenaban los resultados del proceso (lp)
  const Neigh *fb = (const Neigh*)owlBufferGetPointer(frameBuffer,0);

  // Número de intersecciones
  long long intersections = 0;
  for(int j=0; j<nsearchpoints; j++){
    intersections += fb[j*knn].ind;
  }

  cout<< "Número de intersecciones: "<< intersections<<endl;

  printf("Complete Search, writing output to file...\n");
  std::ofstream outfile;
  outfile.open(argv[5]);


  cout << nsearchpoints << "  ||  " << knn << endl;
  for(int j=0; j<nsearchpoints; j++){
    for(int i = 0; i < knn; i++){
       outfile<<fb[j*knn+i].ind<<'\t'<<fb[j*knn+i].dist<<endl;
     }
   }
  outfile.close();

  // and finally, clean up
  owlContextDestroy(context);

}
