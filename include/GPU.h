#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
    
    PrefixMatch() : matchedTokens(0), blockIds() {}
};

class MemoryBlock;
class Request;
class GenerativeRequest;
class EmbeddingRequest;

class GPU {
public:
    GPU(std::size_t blocksNum, std::size_t blockSize);
    ~GPU();
    GPU(const GPU& other);
    GPU& operator=(const GPU& other);
    GPU(GPU&& other);
    GPU& operator=(GPU&& other);
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    std::size_t getBlockSize() const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;
    int getCacheHits() const;
    int getEvictions() const;
    std::size_t getBlocksNum() const;
private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;
    std::size_t totalReserved = 0;
    int cacheHits = 0;
    int evictions = 0;
    std::vector<std::string> slotModelIds;
    std::vector<std::vector<std::string>> slotPrefixKeys;
};

#endif