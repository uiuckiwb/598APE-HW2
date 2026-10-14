#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <node.h>

namespace genetic {
namespace detail {

static constexpr float MIN_VAL = 0.001f;

inline bool is_terminal(node::type t) {
  return t == node::type::variable || t == node::type::constant;
}

inline bool is_nonterminal(node::type t) { return !is_terminal(t); }

inline int arity(node::type t) {
  if (node::type::unary_begin <= t && t <= node::type::unary_end) {
    return 1;
  }
  if (node::type::binary_begin <= t && t <= node::type::binary_end) {
    return 2;
  }
  return 0;
}

// `data` assumed to be stored in col-major format
inline float evaluate_node(const node &n, const float *data, const uint64_t stride,
                    const uint64_t idx, const float *in) {
  if (n.t == node::type::constant) {
    return n.u.val;
  } else if (n.t == node::type::variable) {
    return data[(stride * n.u.fid) + idx];
  } else {
    auto abs_inval = fabsf(in[0]), abs_inval1 = fabsf(in[1]);
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
    // binary operators
    case node::type::add:
      return in[0] + in[1];
    case node::type::atan2:
      return atan2f(in[0], in[1]);
    case node::type::div:
      return abs_inval1 < MIN_VAL ? 1.0f : (in[0]/in[1]);//fdividef(in[0], in[1]);
    case node::type::fdim:
      return fdimf(in[0], in[1]);
    case node::type::gt:
      return in[0] > in[1] ? 1.0f : 0.0f;
    case node::type::max:
      return fmaxf(in[0], in[1]);
    case node::type::min:
      return fminf(in[0], in[1]);
    case node::type::mul:
      return in[0] * in[1];
    case node::type::pow:
      return powf(in[0], in[1]);
    case node::type::sub:
      return in[0] - in[1];
    // unary operators
    case node::type::abs:
      return abs_inval;
    case node::type::acos:
      return acosf(in[0]);
    case node::type::acosh:
      return acoshf(in[0]);
    case node::type::asin:
      return asinf(in[0]);
    case node::type::asinh:
      return asinhf(in[0]);
    case node::type::atan:
      return atanf(in[0]);
    case node::type::atanh:
      return atanhf(in[0]);
    case node::type::cbrt:
      return cbrtf(in[0]);
    case node::type::cos:
      return cosf(in[0]);
    case node::type::cosh:
      return coshf(in[0]);
    case node::type::cube:
      return in[0] * in[0] * in[0];
    case node::type::exp:
      return expf(in[0]);
    case node::type::inv:
      return abs_inval < MIN_VAL ? 0.f : 1.f / in[0];
    case node::type::log:
      return abs_inval < MIN_VAL ? 0.f : logf(abs_inval);
    case node::type::neg:
      return -in[0];
    case node::type::rcbrt:
      return static_cast<float>(1.0) / cbrtf(in[0]);
    case node::type::rsqrt:
      return static_cast<float>(1.0) / sqrtf(abs_inval);
    case node::type::sin:
      return sinf(in[0]);
    case node::type::sinh:
      return sinhf(in[0]);
    case node::type::sq:
      return in[0] * in[0];
    case node::type::sqrt:
      return sqrtf(abs_inval);
    case node::type::tan:
      return tanf(in[0]);
    case node::type::tanh:
      return tanhf(in[0]);
    // shouldn't reach here!
    default:
      return 0.f;
    };
  }
}


inline void evaluate_node_vec(const node &n, const float *data, const uint64_t stride,
			const uint64_t row0,
			const uint64_t m, 
			const float *a,
			const float *b, 
			float *dst) {
  if (n.t == node::type::constant) {
    const float v = n.u.val;
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = v;
    return;
  } else if (n.t == node::type::variable) {
    const float *col = data + (stride * n.u.fid) + row0;
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = col[i];
    return;
  } else {
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
    // binary operators

    case node::type::add:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = a[i] + b[i];
    return;
    case node::type::atan2:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = atan2f(a[i], b[i]);
    return;
    case node::type::div:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i)
      dst[i] = fabsf(b[i]) < MIN_VAL ? 1.0f : (a[i] / b[i]);
    return;
    case node::type::fdim:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = fdimf(a[i], b[i]);
    return;
    case node::type::gt:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = a[i] > b[i] ? 1.0f : 0.0f;
    return;
    case node::type::max:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = fmaxf(a[i], b[i]);
    return;
    case node::type::min:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = fminf(a[i], b[i]);
    return;
    case node::type::mul:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = a[i] * b[i];
    return;
    case node::type::pow:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = powf(a[i], b[i]);
    return;
    case node::type::sub:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = a[i] - b[i];
    return;

    // unary operators


    case node::type::abs:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = fabsf(a[i]);
    return;
    case node::type::acos:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = acosf(a[i]);
    return;
    case node::type::acosh:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = acoshf(a[i]);
    return;
    case node::type::asin:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = asinf(a[i]);
    return;
    case node::type::asinh:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = asinhf(a[i]);
    return;
    case node::type::atan:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = atanf(a[i]);
    return;
    case node::type::atanh:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = atanhf(a[i]);
    return;
    case node::type::cbrt:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = cbrtf(a[i]);
    return;
    case node::type::cos:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = cosf(a[i]);
    return;
    case node::type::cosh:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = coshf(a[i]);
    return;
    case node::type::cube:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = a[i] * a[i] * a[i];
    return;
    case node::type::exp:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = expf(a[i]);
    return;
    case node::type::inv:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i)
      dst[i] = fabsf(a[i]) < MIN_VAL ? 0.0f : 1.0f / a[i];
    return;
    case node::type::log:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) {
      const float av = fabsf(a[i]);
      dst[i] = av < MIN_VAL ? 0.0f : logf(av);
    }
    return;
    case node::type::neg:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = -a[i];
    return;
    case node::type::rcbrt:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = 1.0f / cbrtf(a[i]);
    return;
    case node::type::rsqrt:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = 1.0f / sqrtf(fabsf(a[i]));
    return;
    case node::type::sin:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = sinf(a[i]);
    return;
    case node::type::sinh:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = sinhf(a[i]);
    return;
    case node::type::sq:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = a[i] * a[i];
    return;
    case node::type::sqrt:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = sqrtf(fabsf(a[i]));
    return;
    case node::type::tan:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = tanf(a[i]);
    return;
    case node::type::tanh:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = tanhf(a[i]);
    return;

    // shouldn't reach here!
    default:
    #pragma omp simd
    for (uint64_t i = 0; i < m; ++i) dst[i] = 0.f;
    return;
    };
  }
}




} // namespace detail
} // namespace genetic
