#pragma once
#include "hnswlib.h"
#include "utils/dist_func.h"

namespace hnswlib {

#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
static float AbsInnerProductSIMD(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return fabs(utils::InnerProductFloatAVX512(pVec1v, pVec2v, dim_ptr));
    #elif defined(USE_AVX)
        return fabs(utils::InnerProductFloatAVX(pVec1v, pVec2v, dim_ptr));
    #elif defined(USE_SSE)
        return fabs(utils::InnerProductFloatSSE(pVec1v, pVec2v, dim_ptr));
    #endif
}
#endif

class AbsMinInnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    AbsMinInnerProductSpace(size_t dim) {
        fstdistfunc_ = utils::AbsInnerProduct;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
        fstdistfunc_ = AbsInnerProductSIMD;
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

    ~AbsMinInnerProductSpace() {}
};
}