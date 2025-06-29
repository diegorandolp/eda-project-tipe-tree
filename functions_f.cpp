#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ArkadeModel.h"


using namespace std; 
namespace py = pybind11;


string reverse_string(const string& input) {
    return string(input.rbegin(), input.rend());
    
}

PYBIND11_MODULE(functions_f, m){
    handle.def("reverse_string", &reverse_string, "Invierte una cadena de texto");

    py::class_<ArkadeModel>(m, "ArkadeModel")
        .def(py::init<std::string, std::string, float, int, int, int, std::string>(),
             py::arg("dataPath"),
             py::arg("distance"),
             py::arg("radio"),
             py::arg("k"),
             py::arg("num_data_points"),
             py::arg("num_search"),
             py::arg("outputPath"))
        ;
}