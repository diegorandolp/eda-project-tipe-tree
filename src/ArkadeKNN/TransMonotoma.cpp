#include "TransMonotoma.h"

TransMonotoma::TransMonotoma(const string& dist) : distance_type(dist){

    // transformación monótoma del coseno.
    if(distance_type == "coseno")
        funct = [](vec3f v) -> vec3f{
            return methods::normalize(v);
        };

    // Todas las dempas que devuelvan lo mismo.
    else
        funct = [](vec3f v) -> vec3f{
            return v;
        };

}

vec3f TransMonotoma::transformar(owl::common::vec3f v) {
    return funct(v);
}