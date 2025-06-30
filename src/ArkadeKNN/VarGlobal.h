//
// Created by RODRIGO on 20/06/2025.
//

#ifndef OPTIX_OWL_VARGLOBAL_H
#define OPTIX_OWL_VARGLOBAL_H

#include "DeviceCode.h"
#include <iostream>
#include <cctype>
#include <string>
#include <vector>
#include "TransMonotoma.h"

extern vector<EDA::Point> DataPoints;
extern vector<EDA::Point> QueryPoints;
extern vector<EDA::Neigh> Vecinos;
extern TransMonotoma* trans;
extern int NORM;

extern "C" char DeviceCode_ptx[];

int NormToUse(string& distance);

bool esEntero(const string& str);

#endif //OPTIX_OWL_VARGLOBAL_H
