#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include "heap.h"


namespace hnswlib {
    template<typename dist_t>
    float get_recall_by_val(const std::size_t kQueryNum, const std::size_t kGtSize, const dist_t *kGtVal, const unsigned topk,
                            std::vector<std::vector<HeapItem<dist_t, unsigned>>> &res) {
        float result = 0;
        for (std::size_t q=0; q<kQueryNum; ++q) {
            float gt_val = kGtVal[q*kGtSize+topk-1];
            for (std::size_t k=topk; k>0; --k) {
                if (res[q][k-1].dist_ <= gt_val) {
                    result += (float)k / topk;
                    break;
                }
            }
        }
        return result / kQueryNum;
    }


    template<typename dist_t>
    float get_recall_by_val(const std::size_t kQueryNum, const std::size_t kGtSize, const dist_t *kGtVal,
                            std::vector<std::priority_queue<std::pair<dist_t, std::size_t>>> &res) {
        float correct_num = 0;
        std::size_t r_size = res[0].size();
        for (std::size_t q=0; q<kQueryNum; ++q) {
            std::vector<bool> flag(r_size, true);
            for (; !res[q].empty(); res[q].pop()) {
                const auto& r=res[q].top();
                for (std::size_t i=0; i<r_size; ++i) {
                    if (flag[i] && fabs(r.first - kGtVal[i])<1e-4) {
                        ++correct_num;
                        flag[i] = false;
                        break;
                    }
                }
            }
            kGtVal += kGtSize;
        }
        return correct_num / (kQueryNum*r_size);
    }


    template<typename dist_t>
    float get_recall_by_id(const std::size_t kQueryNum, const std::size_t kGtSize, const unsigned *kGtIds,
                           std::vector<std::priority_queue<std::pair<dist_t, std::size_t>>> &res) {
        float correct_num = 0;
        std::size_t r_size = res[0].size();
        for (std::size_t q=0; q<kQueryNum; ++q) {
            std::vector<bool> flag(r_size, true);
            for (; !res[q].empty(); res[q].pop()) {
                const auto& r=res[q].top();
                for (std::size_t i=0; i<r_size; ++i) {
                    if (flag[i] && r.second == kGtIds[i]) {
                        ++correct_num;
                        flag[i] = false;
                        break;
                    }
                }
            }
            kGtIds += kGtSize;
        }
        return correct_num / (kQueryNum*r_size);
    }

}