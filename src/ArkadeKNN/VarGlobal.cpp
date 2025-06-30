#include "VarGlobal.h"

vector<EDA::Point> DataPoints;
vector<EDA::Point> QueryPoints;
vector<EDA::Neigh> Vecinos;
TransMonotoma* trans;

bool esEntero(const string& str) {

    if (str.empty()) return false;

    size_t i = 0;

    if(str[0] == '-'){
        cerr << "No existe la norma negativa (fuentes: EDA).\n";
        exit(0);
    }

    if (str[0] == '+') {
        if (str.length() == 1) return false;
        i = 1;
    }

    for (; i < str.size(); ++i)
        if (!isdigit(str[i]))
            return false;


    return true;
}
int NormToUse(string& distance){

    for(char& c : distance)
        c = tolower(c);

    int norma;

    if(distance == "euclidian" || distance == "euclidiana" || distance == "2" || distance == "mahalanobis" ||
    distance == "coseno" || distance == "angular")
        norma = 2;
    else if(distance == "manhattan" || distance == "1" || distance == "hamming")
        norma = 1;
    else if(distance == "ceviche" || distance == "chebyshov" || distance == "0")
        norma = 0;
    else if(esEntero(distance))
        norma = stoi(distance);
    else{
        cerr << "La distancia " << distance << " se escribió mal o no es soportada por Arkade\n";
        cout << "Distancias soportadas: euclidian, manhattan, chebyshov, hamming, mahalanobis, coseno y angular\n";
        exit(0);
    }

    return norma;

}