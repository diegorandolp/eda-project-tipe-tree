#include "VarGlobal.h"
#include "ArkadeModel.h"
#include "FastRNN.h"

int main(int ac, char **argv){

    string dataPath = argv[1];
    float radio = stof(argv[4]);
    int k = KN;
    int num_data_points = stoi(argv[2]);
    int num_search = stoi(argv[3]);
    string outputPath = argv[5];
    bool withTrueKnn = false;

    string distance = "0";

    //ArkadeModel model(dataPath, distance, radio, k, num_data_points, num_search, withTrueKnn, outputPath);
    //FastRNN model(dataPath, distance, radio, k, num_data_points, num_search, withTrueKnn, outputPath);

    //cout << model;

}