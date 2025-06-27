//
// Created by RODRIGO on 21/06/2025.
//

#ifndef OPTIX_OWL_TRANSMONOTOMA_H
#define OPTIX_OWL_TRANSMONOTOMA_H

#include <functional>
#include "methods.h"
#include <string>

class TransMonotoma {
private:
    function<vec3f(vec3f)> funct;
    string distance_type;
public:

     explicit TransMonotoma(const string& dist);

     vec3f transformar(vec3f v);

};


#endif //OPTIX_OWL_TRANSMONOTOMA_H
