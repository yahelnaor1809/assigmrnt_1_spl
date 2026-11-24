#include "SchedulingPolicy.h"
#include <algorithm>
#include "Request.h"
#include "GPU.h"
#include "Model.h"
#include <algorithm>

SchedulingPolicy* ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}

std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<int> indices(eligible.size());
    for (std::size_t i = 0; i < eligible.size(); ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&eligible](int a, int b) {
        if (eligible[a]->getAdmissionCycle() != eligible[b]->getAdmissionCycle()) {
            return eligible[a]->getAdmissionCycle() < eligible[b]->getAdmissionCycle();
        }
        return eligible[a]->getId() < eligible[b]->getId();
    });

    return indices;
}

SchedulingPolicy* ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<int> indices(eligible.size());
    for (std::size_t i = 0; i < eligible.size(); ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&eligible, &gpus](int a, int b) {
        std::size_t blocksA = gpus[0].getBlocksForMemory(eligible[a]->getMaxNeededMemory());
        std::size_t blocksB = gpus[0].getBlocksForMemory(eligible[b]->getMaxNeededMemory());
        
        if (blocksA != blocksB) {
            return blocksA < blocksB;
        }
        return eligible[a]->getId() < eligible[b]->getId();
    });

    return indices;
}

SchedulingPolicy* LongestPrefixPolicy::clone() const {
    return new LongestPrefixPolicy(*this);
}

std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<int> indices(eligible.size());
    std::vector<std::size_t> max_h(eligible.size(), 0);

    for (std::size_t i = 0; i < eligible.size(); ++i) {
        indices[i] = i;
        const Request* req = eligible[i];
        std::vector<std::string> cacheableTokens = req->getCacheableTokens();
        
        if (cacheableTokens.empty()) {
            max_h[i] = 0;
            continue;
        }

        std::size_t best_h = 0;
        for (const GPU& gpu : gpus) {
            PrefixMatch match = gpu.findLongestCachedPrefix(req->getModel().getId(), req->getModel().getEmbeddingSize(), cacheableTokens);
            if (match.matchedTokens > best_h) {
                best_h = match.matchedTokens;
            }
        }
        max_h[i] = best_h;
    }

    std::sort(indices.begin(), indices.end(), [&eligible, &max_h](int a, int b) {
        if (max_h[a] != max_h[b]) {
            return max_h[a] > max_h[b];
        }
        return eligible[a]->getId() < eligible[b]->getId();
    });

    return indices;
}