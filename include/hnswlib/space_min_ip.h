#pragma once
#include "hnswlib.h"
#include "utils/dist_func.h"

namespace hnswlib {

class MinInnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    MinInnerProductSpace(size_t dim) {
        fstdistfunc_ = utils::InnerProduct;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        fstdistfunc_ = utils::InnerProductFloatAVX512;
    #elif defined(USE_AVX)
        fstdistfunc_ = utils::InnerProductFloatAVX;
    #elif defined(USE_SSE)
        fstdistfunc_ = utils::InnerProductFloatSSE;
    #endif
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

    ~MinInnerProductSpace() {}
};

}  // namespace hnswlib