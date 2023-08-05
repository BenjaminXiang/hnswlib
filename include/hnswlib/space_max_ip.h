#pragma once
#include "hnswlib.h"
#include "utils/dist_func.h"

namespace hnswlib {
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
static float InverseInnerProductSIMD(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return -utils::InnerProductFloatAVX512(pVec1v, pVec2v, dim_ptr);
    #elif defined(USE_AVX)
        return -utils::InnerProductFloatAVX(pVec1v, pVec2v, dim_ptr);
    #elif defined(USE_SSE)
        return -utils::InnerProductFloatSSE(pVec1v, pVec2v, dim_ptr);
    #endif
}
#endif

class MaxInnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    MaxInnerProductSpace(size_t dim) {
        fstdistfunc_ = utils::InverseInnerProduct;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
        fstdistfunc_ = InverseInnerProductSIMD;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~MaxInnerProductSpace() {}
};
}