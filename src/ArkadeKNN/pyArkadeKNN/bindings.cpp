#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../ArkadeModel.h"

namespace py = pybind11;

PYBIND11_MODULE(py_arkade, m) {
    m.doc() = "Python bindings for ArkadeKNN";

    py::class_<ArkadeModel>(m, "ArkadeModel")
        .def(py::init<std::string, std::string, float, int, int, int, std::string>(),
             py::arg("dataPath"),
             py::arg("distance"),
             py::arg("radio"),
             py::arg("k"),
             py::arg("num_data_points"),
             py::arg("num_search"),
             py::arg("outputFile"))
        .def("initialize_data", &ArkadeModel::InitializeData)
        .def("get_radius", &ArkadeModel::get_radius)
        .def("get_k", &ArkadeModel::get_k)
        .def("get_num_search", &ArkadeModel::get_num_search)
        .def("get_distance_type", &ArkadeModel::get_distance_type);
}
