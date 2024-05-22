#pragma once
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>

inline void *alloc2M(size_t nbytes) {
    size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
    auto p = std::aligned_alloc(1 << 21, len);
    std::memset(p, 0, len);
    return p;
}
inline void *alloc64B(size_t nbytes) {
    size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
    auto p = std::aligned_alloc(1 << 6, len);
    std::memset(p, 0, len);
    return p;
}
inline constexpr int64_t do_align(int64_t x, int64_t align) {
    return (x + align - 1) / align * align;
}
static int
IPSqrSQ8(const void *pVect1v, const void *pVect2v, int d) {
    size_t i;
    int8_t *x = (int8_t *) pVect1v;
    int8_t *y = (int8_t *) pVect2v;
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < d; i += 16) {
        __m128i xx = _mm_loadu_si128((__m128i *)(x + i));
        __m128i yy = _mm_loadu_si128((__m128i *)(y + i));
        __m256i xx_ext = _mm256_cvtepi8_epi16(xx);
        __m256i yy_ext = _mm256_cvtepi8_epi16(yy);
        sum = _mm256_add_epi32(_mm256_madd_epi16(xx_ext, yy_ext), sum);
    }
    __m128i sumh = _mm_add_epi32(_mm256_extracti32x4_epi32(sum, 0), _mm256_extracti32x4_epi32(sum, 1));
    __m128i tmp = _mm_hadd_epi32(sumh, sumh);
    return _mm_extract_epi32(tmp, 0) + _mm_extract_epi32(tmp, 1);
}
inline float reduce_add_f32x16(__m512 x) {
  auto sumh =
      _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
  auto sumhh =
      _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}
inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {

#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    auto zz = _mm_loadu_epi8(y + i);
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    sum = _mm512_fmadd_ps(xx, yy, sum);
  }
  return -reduce_add_f32x16(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = y[i] + 0.5f;
    yy = yy * dif[i] + mi[i] * 255.0f;
    sum += x[i] * yy;
  }
  return -sum;
#endif
}
// static int
// IPSqrSQ8(const void *pVect1v, const void *pVect2v, int d) {
//     size_t i;
//     int8_t *x = (int8_t *) pVect1v;
//     int8_t *y = (int8_t *) pVect2v;
//     int sum = 0;
//     for (int i = 0; i < d; i++) {
//         sum += (int)x[i] * (int)y[i];
//     }
//     return sum;
// }
struct SQ8Quantizer {
  using data_type = int8_t;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;
  float alpha = 0.0f;

  SQ8Quantizer() = default;

  explicit SQ8Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
        alpha(0.0f) {}

  ~SQ8Quantizer() { free(codes); }

  void train(const float *data, int n) {
    for (size_t i = 0; i < n; ++i) {
        const float* vec = data + i * d;
        for (size_t j = 0; j < d; ++j) {
            alpha = std::max(alpha, std::abs(vec[j]));
        }
    }
    // printf("%0.5f\n", alpha);
    codes = (char *)alloc2M((size_t)n * code_size);
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  char *get_data(int u) const { return codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (size_t i = 0; i < d; ++i) {
        float x = from[i] / alpha;
        if (x > 1.0f) {
            x = 1.0f;
        }
        if (x < -1.0f) {
            x = -1.0f;
        }
        to[i] = std::round(x * 127.0f);
        // printf("%0.5f %d\n", from[i], (int)to[i]);
    }
  }

  struct Computer {
    const SQ8Quantizer &quant;
    int8_t *q;
    float alpha;
    Computer(const SQ8Quantizer &quant, const float *query)
        : quant(quant), q((int8_t*)alloc64B(quant.d_align)), alpha(quant.alpha)
           {
        q = (int8_t*)alloc64B(quant.d_align);
        quant.encode(query, (char*)q);
    }
    ~Computer() { free(q); }
    int operator()(int u) const {
      return IPSqrSQ8(q, (data_type *) quant.get_data(u), quant.d_align);
    }
  };
  auto get_computer(const float *query) const {
    return Computer(*this, query);
  }
  // using data_type = uint8_t;
  // constexpr static int kAlign = 16;
  // int d, d_align;
  // int64_t code_size;
  // char *codes = nullptr;
  // std::vector<float> mx, mi, dif;

  // SQ8Quantizer() = default;

  // explicit SQ8Quantizer(int dim)
  //     : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
  //       mx(d_align, -HUGE_VALF), mi(d_align, HUGE_VALF), dif(d_align) {}

  // ~SQ8Quantizer() { free(codes); }

  // void train(const float *data, int n) {
  //   for (int64_t i = 0; i < n; ++i) {
  //     for (int64_t j = 0; j < d; ++j) {
  //       mx[j] = std::max(mx[j], data[i * d + j]);
  //       mi[j] = std::min(mi[j], data[i * d + j]);
  //     }
  //   }
  //   for (int64_t j = 0; j < d; ++j) {
  //     dif[j] = mx[j] - mi[j];
  //   }
  //   for (int64_t j = d; j < d_align; ++j) {
  //     dif[j] = mx[j] = mi[j] = 0;
  //   }
  //   codes = (char *)alloc2M((size_t)n * code_size);
  //   for (int i = 0; i < n; ++i) {
  //     encode(data + i * d, get_data(i));
  //   }
  //   // printf("%0.5f %0.5f\n", mx[d - 1], mi[d - 1]);
  // }

  // char *get_data(int u) const { return codes + u * code_size; }

  // void encode(const float *from, char *to) const {
  //   for (int j = 0; j < d; ++j) {
  //     float x = (from[j] - mi[j]) / dif[j];
  //     if (x < 0.0) {
  //       x = 0.0;
  //     }
  //     if (x > 1.0) {
  //       x = 1.0;
  //     }
  //     uint8_t y = x * 255;
  //     to[j] = y;
  //   }
  // }

  // struct Computer {
  //   using dist_type = float;
  //   constexpr static auto dist_func = IPSQ8_ext;
  //   const SQ8Quantizer &quant;
  //   float *q;
  //   const float *mi, *dif;
  //   Computer(const SQ8Quantizer &quant, const float *query)
  //       : quant(quant), q((float *)alloc64B(quant.d_align * 4)),
  //         mi(quant.mi.data()), dif(quant.dif.data()) {
  //     std::memcpy(q, query, quant.d * 4);
  //   }
  //   ~Computer() { free(q); }
  //   dist_type operator()(int u) const {
  //     return dist_func(q, (data_type *)quant.get_data(u), quant.d_align, mi,
  //                      dif);
  //   }
  // };

  // auto get_computer(const float *query) const {
  //   return Computer(*this, query);
  // }
};