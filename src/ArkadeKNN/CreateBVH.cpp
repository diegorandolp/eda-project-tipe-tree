#include "CreateBVH.h"

#include <utility>
#include "BaseModel.h"

// Checar esto, posiblemente falle.
CreateBVH::~CreateBVH() {
    owlContextDestroy(context);
}

CreateBVH::CreateBVH(BaseModel* _model, bool _isTrueKnnModel, string _distance) {


    cout << " \n\t ==================== Configuracion de mi Dispositivo ==================== \n\n";

    isTrueKnnModel = _isTrueKnnModel;
    distance = std::move(_distance);
    context = owlContextCreate(nullptr, 1);
    module  = owlModuleCreate(context, DeviceCode_ptx);

    model = _model;
    model->InitializeData();

    n_neighbors.resize(model->get_num_search(), 0);
    results = GetResultsAndCreateBVH();
}

// Se podría dividir todo esto en varios métodos pero debería guardar muchos atributos en la clase xd
// lo hago otro día si estoy aburrido
const EDA::Neigh* CreateBVH::GetResultsAndCreateBVH() {

    cout << " \n\t ==================== Configuracion y creacion del BVH ==================== \n\n";

    // Instancia de OWL con la clase personalizada SpheresGeom, para el uso de geometrías.
    OWLVarDecl SpheresGeomVars[] = {
            { "data_pts",  OWL_BUFPTR, OWL_OFFSETOF(EDA::SpheresGeom, data_spheres)},
            { "rad", OWL_FLOAT, OWL_OFFSETOF(EDA::SpheresGeom, rad)},
            { /* sentinel to mark end of list */ }
    };

    // Permite personalizar la geometría de intersección de los rayos, colocandolas en el contexto creado,
    // especificando que es una geometría del usuario y el tamaño de bytes de la clase, como su instancia en optix.
    OWLGeomType SpheresGeomType = owlGeomTypeCreate(context,
                                OWL_GEOMETRY_USER,
                                sizeof(EDA::SpheresGeom),
                                SpheresGeomVars,-1);

    // En el módulo creado, designa el tipo de intersecciones que tendrán los rayos con las geometrías especificadas.
    owlGeomTypeSetIntersectProg(SpheresGeomType,0,
                                module,"Spheres");

    // Permite especificar la geometría que el AABB (nodo del BVH) va a envolver en el arbol, en el módulo especificado.
    owlGeomTypeSetBoundsProg(SpheresGeomType,
                             module,"Spheres");

    // Con la configuración previa del módulo y contexto, configura el proceso de construcción del BVH (Aún no lo crea).
    owlBuildPrograms(context);

    // Buffer en el que se rescataran y guardaran los resultados del KNN, en un espacio en memoria de la GPU.
    // Se especifica el contexto, el array de vecinos como su cantidad esperada.
    OWLBuffer frameBuffer = owlHostPinnedBufferCreate(context, OWL_USER_TYPE(Vecinos[0]),
                                                      Vecinos.size());

    // Para el TrueKNN
    OWLBuffer n_neighBuffer
            = owlHostPinnedBufferCreate(context, OWL_INT,
                                        n_neighbors.size());

    owlBufferUpload(n_neighBuffer, n_neighbors.data());

    // Se crean buffers que almacenen todo el contenido de la Data y la Query, para poder usarlas en la GPU.
    // Se debe especificar el contexto, el type personalizado, el tamaño y la cantidad de datos, del mismo type, a utilizar.
    // Esto permitirá llegar toda nuestra fuente de datos extraida en los vectores a su uso en la GPU.
    OWLBuffer DataPointsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(DataPoints[0]),
                                    DataPoints.size(), DataPoints.data());

    OWLBuffer QueryPointsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(QueryPoints[0]),
                                    QueryPoints.size(), QueryPoints.data());

    // Permite crear las esferas a utilizar en el contexto especificado
    // En esta parte se llevará los data points extraídos del data set a un entorno manejable
    // por owl, configurando el tipado, el tamaño a utilizar y los valores a utilizar.

    // Esto servirá para contener todas las geometrías primitivas definidas en el buffer DataPointsBuffer
    OWLGeom SpheresGeom = owlGeomCreate(context, SpheresGeomType);

    // Seteamos el número de datos que se van a usar.
    // Internamente etiqueta a cada geometría según su índice en el vector, así que cada
    // optixGetPrimitiveIndex() representa la fila del txt original, al que pertenece ese punto.
    owlGeomSetPrimCount(SpheresGeom, DataPoints.size());

    // Seteamos todo el conjunto de data points en el objeto owl SpheresGeom, que contiene la configuración
    // para contener las geometrías de la data, contenido en el atributo data_pts especificado en SpheresGeomVars.
    owlGeomSetBuffer(SpheresGeom, "data_pts", DataPointsBuffer);

    // Seteamos el radio que tendrá cada geometría, contenido en el atributo rad especificado en SpheresGeomVars.
    owlGeomSet1f(SpheresGeom, "rad", model->get_radius());

    // Para el TrueKNN: Contador del número de rondas en las que se expandió el radio.
    round = 0;
    bool isTruKnn = isTrueKnnModel;

    // Crear un objeto OWL que contenga la estructura personalizada de las variables globales a utilizar
    // en este caso sería el frameBuffer de los resultados y los k vecinos a buscar.
    OWLVarDecl myGlobalsVars[] = {
            {"frameBuffer", OWL_BUFPTR, OWL_OFFSETOF(EDA::GlobalVars, frameBuffer)},
            {"k", OWL_INT, OWL_OFFSETOF(EDA::GlobalVars, k)},
            {"NORM", OWL_INT, OWL_OFFSETOF(EDA::GlobalVars, NORM)},
            {"round", OWL_INT, OWL_OFFSETOF(EDA::GlobalVars, round)},
            {"isTrueKnn", OWL_BOOL, OWL_OFFSETOF(EDA::GlobalVars, isTrueKnn)},
            {"num_neighbors", OWL_BUFPTR, OWL_OFFSETOF(EDA::GlobalVars, num_neighbors)},
            { /* sentinel to mark end of list */ }
    };

    OWLParams lp = owlParamsCreate(context, sizeof(EDA::GlobalVars), myGlobalsVars, -1);

    // Seteamos los valores...
    owlParamsSetBuffer(lp, "frameBuffer", frameBuffer);
    owlParamsSet1i(lp, "k", model->get_k());
    cout << "NORMA::: " << NormToUse(distance);
    owlParamsSet1i(lp, "NORM", NormToUse(distance));
    owlParamsSet1i(lp, "round", round);
    owlParamsSet1b(lp, "isTrueKnn", isTruKnn);
    owlParamsSetBuffer(lp, "num_neighbors", n_neighBuffer);


    // Guardamos todas los conjuntos de geometrías a utilizar, en nuestro caso solamente de los data points em su formato OWL.
    OWLGeom  userGeoms[] = {
            SpheresGeom
    };

    // Acá empezaremos con la construcción del BVH, midiendo el tiempo que tarda en construirse.
    auto start_b = std::chrono::steady_clock::now();

    // Construimos el escenario que contengan todas las diferentes conjuntos de geometrías, en nuestro caso
    // solamente el grupo de los data points en su formato OWL en el espacio del contexto colocado.

    // la etiqueta: OPTIX_BUILD_FLAG_ALLOW_UPDATE, permite actualizar en tiempo de ejecución las geometrías (para el TrueKnn)
    OWLGroup spheresGroup = owlUserGeomGroupCreate(context, 1, userGeoms, OPTIX_BUILD_FLAG_ALLOW_UPDATE);


    // Con este método, una vez configurado lo anterior, se construye el BVH con el grupo de geometrías especificadas
    // Internamente, para su construcción, para poder englobar una geometría en un AABB se activa el macro
    // OPTIX_BOUNDS_PROGRAM, el cual es personalizado por el usuario para poder crear el BVH especificando que tanto
    // quiero que englobe un AABB al momento de englobar una geometría.

    // Esto solamente permite tener el escenario listo, englobado y podado por el AABB en las geometrías
    owlGroupBuildAccel(spheresGroup);

    // Se crea una instancia del escenario para poder crear el árbol BVH
    OWLGroup world = owlInstanceGroupCreate(context, 1, &spheresGroup, nullptr, nullptr,
                                            OWL_MATRIX_FORMAT_OWL,
                                            OPTIX_BUILD_FLAG_ALLOW_UPDATE);

    // Construimos el árbol
    owlGroupBuildAccel(world);

    auto end_b = std::chrono::steady_clock::now();

    auto elapsed_b = std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
    TimeCreateBVH = elapsed_b.count()/1000000.0;



    cout << " \n\t ==================== Lanzamiento de Rayos (Query) ==================== \n\n";

    // Creamos la estructura personalizada de los rayos que se van a lanzar en el BVH tree
    OWLVarDecl rayGenVars[] = {
            {"query_pts",       OWL_BUFPTR, OWL_OFFSETOF(EDA::RayGenData, query_pts)},
            { "world",         OWL_GROUP,  OWL_OFFSETOF(EDA::RayGenData, world)},
            { /* sentinel to mark end of list */ }
    };


    // Configuramos la variable OWL de rayGenVars como el objeto que se encargará de lanzar
    // los rayos al BVH tree.
    OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen",
                              sizeof(EDA::RayGenData),
                              rayGenVars, -1);

    // Seteamos los atributos del objeto OWL
    owlRayGenSetBuffer(rayGen, "query_pts", QueryPointsBuffer);
    owlRayGenSetGroup (rayGen, "world", world);

    // Compila las funciones del Optix, construyendo el escenario.
    owlBuildPrograms(context);

    // Construye la estructura interna para que OWL pueda realizar el proceso.
    owlBuildPipeline(context);

    // Permite la diferenciación de objetos diferentes creados/configurados, mediante funciones en específico para c/u
    // para que realice y facilite los procesos en la GPU con esos objetos.
    owlBuildSBT(context);

    // Parámetros de owlLaunch2D()
    // raygen: en base a OPTIX_RAYGEN_PROGRAM, se puede acceder al llamado de rayo y ejecutar la acción con CUDA.
    // nsearchpoints: Número de rayos que estoy lanzando en total, que sería igual al número de puntos de la query
    // un rayo por query point.
    // 1: ...
    // lp: variable global owl, donde se guardarán los resultados.

    // Cómo es el recorrido del arbol con la query (rayos), se activa el macro de OPTIX_INTERSECT_PROGRAM
    // que se encarga de evaluar el evento de intersección con un AABB.

    // Acá empezaremos con el trazado de rayos en el BVH, midiendo los tiempos del proceso.
    auto start = std::chrono::steady_clock::now();

    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

    if (err != cudaSuccess) {
        printf("Error al obtener la memoria: %s\n", cudaGetErrorString(err));
    } else {
        printf("Memoria libre: %zu MB\n", free_mem / (1024 * 1024));
        printf("Memoria total: %zu MB\n", total_mem / (1024 * 1024));
    }

    if(isTrueKnnModel){

        bool repeat = true;
        int k = model->get_k();

        // Repetimos el proceso hasta que cada query point tenga mínimamente k vecinos.
        while(repeat){

            owlLaunch2D(rayGen, model->get_num_search(), 1, lp);

            // Espera a que todos los rayos terminen para continuar (porsiaca)
            //cudaDeviceSynchronize();

            //cout << "\n-----------------\n";

            // Acá cuantificamos el # de vecinos por query point.
            const int *nb = (const int*)owlBufferGetPointer(n_neighBuffer, 0);
            repeat = false;

            for(int q = 0; q < model->get_num_search(); q+=1){
                //cout << nb[q] << endl;
                if(nb[q] >= k)
                    continue;

                repeat = true;
                round+=1;
                owlParamsSet1i(lp, "round", round);

                // Actualizamos el rádio para cubrir más data points
                float actual_radio = model->get_radius();
                model->set_radius(actual_radio * 2);

                // Actualizamos el tamaño del radio en los objetos OWL
                owlGeomSet1f(SpheresGeom, "rad", model->get_radius());

                // Al actualizar el radio, también debemos actualizar el tamaño del bounding box que lo cubre.
                owlGroupRefitAccel(spheresGroup);
                owlGroupRefitAccel(world);
                cout << "Ronda: " << round << " --> " << "new radio: " << model->get_radius() << endl;

                break;
            }

        }
    }
    else {

        owlLaunch2D(rayGen, model->get_num_search(), 1, lp);
    }


    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    TimeRayGen = elapsed.count()/1000000.0;

    // Rescatamos los resultados...
    const auto* fb = (const EDA::Neigh*)owlBufferGetPointer(frameBuffer, 0);

    return fb;

}

const EDA::Neigh *CreateBVH::get_results() {
    return results;
}

float CreateBVH::GetTimeCreateBVH() {
    return TimeCreateBVH;
}

float CreateBVH::GetTimeRayGen() {
    return TimeRayGen;
}

int CreateBVH::GetNumRounds() {
    return round;
}

bool CreateBVH::isTrueKNN(){
    return isTrueKnnModel;
}