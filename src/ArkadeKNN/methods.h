//
// Created by RODRIGO on 21/06/2025.
//

#ifndef OPTIX_OWL_METHODS_H
#define OPTIX_OWL_METHODS_H

#include <owl/common/math/random.h>
using namespace std;
using namespace owl;

struct methods{

    // Para la distancia coseno.
    static inline float magnitude(const vec3f& v){
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    static inline vec3f normalize(const vec3f& v){
        float len = magnitude(v);
        if (len == 0.0f) return v;
        return vec3f(v.x / len, v.y / len, v.z / len);
    }

};

#endif //OPTIX_OWL_METHODS_H
