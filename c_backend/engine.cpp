#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "run.h"

namespace py = pybind11;

PYBIND11_MODULE(engine, m) {
    m.doc() = "C++ Backend Engine";
    m.def("run", &run, "Run Algorithm");
    m.def("run2", &run2, "Run Algorithm 2");
    m.def("run3", &run3, "Run Algorithm 3");
}