#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../ArkadeModel.h"
#include "../FastRNN.h"

namespace py = pybind11;

PYBIND11_MODULE(py_arkade, m) {
    m.doc() = "Python bindings for ArkadeKNN";

    py::class_<ArkadeModel>(m, "ArkadeModel")
        .def(py::init<std::string, std::string, float, int, int, int, bool, std::string, bool, std::vector<float>>(),
             py::arg("dataPath"),
             py::arg("distance"),
             py::arg("radio"),
             py::arg("k"),
             py::arg("num_data_points"),
             py::arg("num_search"),
             py::arg("TrueKnn"),
             py::arg("outputFile"),
             py::arg("fromUser") = false,
             py::arg("myInput") = std::vector<float>{});

    py::class_<FastRNN>(m, "FastRNN")
        .def(py::init<std::string, std::string, float, int, int, int, bool, std::string, bool, std::vector<float>>(),
                py::arg("dataPath"),
                py::arg("distance"),
                py::arg("radio"),
                py::arg("k"),
                py::arg("num_data_points"),
                py::arg("num_search"),
                py::arg("TrueKnn"),
                py::arg("outputFile"),
                py::arg("fromUser") = false,
                py::arg("myInput") = std::vector<float>{});
}
