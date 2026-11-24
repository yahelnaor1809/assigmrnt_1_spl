#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr), 
      totalReserved(0), cacheHits(0), evictions(0),
      slotModelIds(blocksNum, ""), slotPrefixKeys(blocksNum, std::vector<std::string>()) {
}

GPU::~GPU() {
    for (std::size_t i = 0; i < memory.size(); ++i) {
        if (memory[i] != nullptr) {
            delete memory[i];
            memory[i] = nullptr;
        }
    }
}

GPU::GPU(const GPU& other)
    : blocksNum(other.blocksNum), blockSize(other.blockSize), memory(other.blocksNum, nullptr), 
      totalReserved(other.totalReserved), cacheHits(other.cacheHits), evictions(other.evictions),
      slotModelIds(other.slotModelIds), slotPrefixKeys(other.slotPrefixKeys) {
    
    for (std::size_t i = 0; i < other.memory.size(); ++i) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            int refs = other.memory[i]->getRefCount(); 
            for (int r = 0; r < refs; ++r) {
                memory[i]->acquire();
            }
        }
    }
}

GPU& GPU::operator=(const GPU& other) {
    if (this != &other) {
        for (std::size_t i = 0; i < memory.size(); ++i) {
            if (memory[i] != nullptr) {
                delete memory[i];
            }
        }
        memory.clear();

        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        totalReserved = other.totalReserved;
        cacheHits = other.cacheHits;
        evictions = other.evictions;
        slotModelIds = other.slotModelIds;
        slotPrefixKeys = other.slotPrefixKeys;
        
        memory.resize(blocksNum, nullptr);

        for (std::size_t i = 0; i < other.memory.size(); ++i) {
            if (other.memory[i] != nullptr) {
                memory[i] = other.memory[i]->clone();
                int refs = other.memory[i]->getRefCount();
                for (int r = 0; r < refs; ++r) {
                    memory[i]->acquire();
                }
            }
        }
    }
    return *this;
}

GPU::GPU(GPU&& other)
    : blocksNum(other.blocksNum), blockSize(other.blockSize), memory(),
      totalReserved(other.totalReserved), cacheHits(other.cacheHits), evictions(other.evictions),
      slotModelIds(std::move(other.slotModelIds)), slotPrefixKeys(std::move(other.slotPrefixKeys)) {
    
    memory.swap(other.memory);
    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
}

GPU& GPU::operator=(GPU&& other) {
    if (this != &other) {
        for (std::size_t i = 0; i < memory.size(); ++i) {
            if (memory[i] != nullptr) {
                delete memory[i];
            }
        }
        
        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        totalReserved = other.totalReserved;
        cacheHits = other.cacheHits;
        evictions = other.evictions;
        slotModelIds = std::move(other.slotModelIds);
        slotPrefixKeys = std::move(other.slotPrefixKeys);
        
        memory.clear();
        memory.swap(other.memory);

        other.blocksNum = 0;
        other.blockSize = 0;
        other.totalReserved = 0;
    }
    return *this;
}

std::size_t GPU::getBlockSize() const {
    return blockSize;
}

int GPU::getCacheHits() const { 
    return cacheHits; 
}

int GPU::getEvictions() const { 
    return evictions; 
}

std::size_t GPU::getBlocksNum() const {
    return blocksNum;
}

std::size_t GPU::availableSlots() const {
    std::size_t activeBlocks = 0;
    for (MemoryBlock* block : memory) {
        if (block != nullptr && block->getRefCount() > 0) {
            activeBlocks++;
        }
    }
    return blocksNum - activeBlocks - totalReserved;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory_needed) const {
    if (memory_needed == 0) return 0;
    return (memory_needed + blockSize - 1) / blockSize;
}

void GPU::reserve(std::size_t n, const Request& request) {
    totalReserved += n;
}

void GPU::unreserve(std::size_t n, const Request& request) {
    if (totalReserved >= n) {
        totalReserved -= n;
    }
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    if (memory[blockId] != nullptr) {
        return memory[blockId]->readFromBlock(offset);
    }
    return 0; 
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    if (memory[blockId] != nullptr) {
        memory[blockId]->writeToBlock(offset, value);
    }
}

std::size_t GPU::allocateBlock(const Request& request) {
    std::size_t selectedSlot = blocksNum;
    
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] == nullptr) {
            selectedSlot = i;
            break;
        }
    }
    
    if (selectedSlot == blocksNum) {
        for (std::size_t i = 0; i < blocksNum; ++i) {
            if (memory[i] != nullptr && memory[i]->getRefCount() == 0) {
                selectedSlot = i;
                delete memory[i];
                memory[i] = nullptr;
                evictions++;
                break;
            }
        }
    }
        
    request.allocateOneBlock(*this, selectedSlot);
    
    if (totalReserved > 0) {
        totalReserved--;
    }
    
    return selectedSlot;
}

void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    memory[slot] = new KVBlock(blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    memory[slot] = new EmbeddingWorkspace(blockSize);
}

void GPU::acquire(std::size_t slot) {
    if (memory[slot] != nullptr) {
        memory[slot]->acquire();
    }
}

void GPU::release(std::size_t slot) {
    if (memory[slot] != nullptr) {
        IdleAction action = memory[slot]->release();
        
        if (memory[slot]->getRefCount() == 0) {
            bool shouldDestroy = false;
            
            if (action == IdleAction::Destroy) {
                shouldDestroy = true;
            } else if (action == IdleAction::KeepCached) {
                if (slotPrefixKeys[slot].empty()) {
                    shouldDestroy = true;
                }
            }
            
            if (shouldDestroy) {
                delete memory[slot];
                memory[slot] = nullptr;
                slotModelIds[slot] = "";
                slotPrefixKeys[slot].clear();
            }
        }
    }
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] != nullptr && slotModelIds[i] == modelId && slotPrefixKeys[i] == prefixTokens) {
            return;
        }
    }
    
    slotModelIds[block] = modelId;
    slotPrefixKeys[block] = prefixTokens;
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize, const std::vector<std::string>& promptTokens) const {
    PrefixMatch match;
    match.matchedTokens = 0;
    
    std::size_t t = blockSize / embeddingSize;
    std::vector<std::string> currentPrefix;
    
    for (std::size_t i = 0; i < promptTokens.size(); i++) {
        currentPrefix.push_back(promptTokens[i]);
        
        if ((i + 1) % t == 0) {
            bool found = false;
            for (std::size_t slot = 0; slot < blocksNum; ++slot) {
                if (memory[slot] != nullptr && slotModelIds[slot] == modelId && slotPrefixKeys[slot] == currentPrefix) {
                    match.blockIds.push_back(slot);
                    match.matchedTokens = i + 1;
                    found = true;
                    break;
                }
            }
            if (!found) break;
        }
    }
    return match;
}

void GPU::copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks) {
    for (std::size_t otherSlot : blocks) {
        bool alreadyExists = false;
        for (std::size_t i = 0; i < blocksNum; ++i) {
            if (memory[i] != nullptr && slotModelIds[i] == other.slotModelIds[otherSlot] && slotPrefixKeys[i] == other.slotPrefixKeys[otherSlot]) {
                alreadyExists = true;
                break;
            }
        }
        if (alreadyExists) continue;

        std::size_t selectedSlot = blocksNum;
        for (std::size_t i = 0; i < blocksNum; ++i) {
            if (memory[i] == nullptr) {
                selectedSlot = i;
                break;
            }
        }

        if (selectedSlot == blocksNum) {
            for (std::size_t i = 0; i < blocksNum; ++i) {
                if (memory[i] != nullptr && memory[i]->getRefCount() == 0) {
                    selectedSlot = i;
                    delete memory[i];
                    memory[i] = nullptr;
                    evictions++;
                    break;
                }
            }
        }

        if (selectedSlot < blocksNum && other.memory[otherSlot] != nullptr) {
            memory[selectedSlot] = other.memory[otherSlot]->clone();
            slotModelIds[selectedSlot] = other.slotModelIds[otherSlot];
            slotPrefixKeys[selectedSlot] = other.slotPrefixKeys[otherSlot];
        }
    }
}