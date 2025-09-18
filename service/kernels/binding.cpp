#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

py::array_t<float>
vec_relu(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    // Expect 1D input; pybind11 will cast/ensure C-contiguous.
    auto a = arr.unchecked<1>();            // throws if not 1D
    const ssize_t n = a.shape(0);

    py::array_t<float> out(n);              // 1D output, same length
    auto b = out.mutable_unchecked<1>();    // writeable view

    for (ssize_t i = 0; i < n; ++i) {
        const float v = a(i);
        b(i) = v > 0.f ? v : 0.f;
    }
    return out; // already correct shape
}

PYBIND11_MODULE(binding, m) {
    m.def("vec_relu", &vec_relu, "Vector ReLU (CPU)");
}
