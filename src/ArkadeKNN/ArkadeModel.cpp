#include "ArkadeModel.h"
#include "CreateBVH.h"
#include <utility>


ArkadeModel::ArkadeModel(std::string dataPath, string _distance, float _radio, int _k, int _num_data_points,
                         int _num_search, bool TrueKnn, std::string outputPath) {

    distance = std::move(_distance);

    // Devolverá el mismo valor si es que no necesita una transformación monotoma (L^p)
    trans = new TransMonotoma(distance);

    baseModel = new BaseModel(std::move(dataPath), _radio, _k, _num_data_points, _num_search, std::move(outputPath));

    bool isTrueKnnModel = TrueKnn;
    auto* gpu_process = new CreateBVH(baseModel, isTrueKnnModel, distance);
    baseModel->create_tree(gpu_process);
    baseModel->obtain_results();

}
