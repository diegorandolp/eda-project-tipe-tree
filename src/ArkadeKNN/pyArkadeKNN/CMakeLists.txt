find_package(PythonLibs 3 REQUIRED)
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../owl/pyOWL/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)

pybind11_add_module(py_arkade
  bindings.cpp
)

target_link_libraries(py_arkade
  PRIVATE
  ArkadeKNN_lib
  Python::Python
  owl::owl
)
