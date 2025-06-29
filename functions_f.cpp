#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std; 

string reverse_string(const string& input) {
    return string(input.rbegin(), input.rend());
    
}



PYBIND11_MODULE(functions_f, handle){
    handle.def("reverse_string", &reverse_string, "Invierte una cadena de texto");
}