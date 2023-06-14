#pragma once
#include "hnswlib.h"
#include "utils/dist_func.h"

namespace hnswlib {

class MinL2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    MinL2Space(size_t dim) {
        fstdistfunc_ = utils::L2Sqr;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        fstdistfunc_ = utils::L2SqrFloatAVX512;
    #elif defined(USE_AVX)
        fstdistfunc_ = utils::L2SqrFloatAVX;
    #elif defined(USE_SSE)
        fstdistfunc_ = utils::L2SqrSSE;
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

    ~MinL2Space() {}
};

}
