#pragma once
#include "hnswlib.h"
#include "utils/dist_func.h"

namespace hnswlib {

#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
static float InverseL2SqrSIMD(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return -utils::L2SqrFloatAVX512(pVec1v, pVec2v, dim_ptr);
    #elif defined(USE_AVX)
        return -utils::L2SqrFloatAVX(pVec1v, pVec2v, dim_ptr);
    #elif defined(USE_SSE)
        return -utils::L2SqrFloatSSE(pVec1v, pVec2v, dim_ptr);
    #endif
}
#endif

class MaxL2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    MaxL2Space(size_t dim) {
        fstdistfunc_ = utils::InverseL2Sqr;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
        fstdistfunc_ = InverseL2SqrSIMD;
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

    ~MaxL2Space() {}
};
}