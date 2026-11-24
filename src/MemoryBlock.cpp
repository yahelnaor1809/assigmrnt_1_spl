#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size) 
    : size(size), refCount(0), data(new int[size]()) {
}

MemoryBlock::~MemoryBlock() {
    delete[] data;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other) 
    : size(other.size), refCount(0), data(new int[other.size]) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = other.data[i];
    }
}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this != &other) {
        delete[] data;
        
        size = other.size;
        refCount = 0;
        data = new int[size];
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
        }
    }
    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) 
    : size(other.size), refCount(other.refCount), data(other.data) {
    other.data = nullptr;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) {
    if (this != &other) {
        delete[] data;
        
        size = other.size;
        refCount = other.refCount;
        data = other.data;
        
        other.data = nullptr;
    }
    return *this;
}
int MemoryBlock:: getRefCount() const {
    return refCount;
}

int MemoryBlock::readFromBlock(std::size_t offset) const {
    return data[offset];
}

void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    data[offset] = value;
}

void MemoryBlock::acquire() {
    refCount++;
}

IdleAction MemoryBlock::release() {
    refCount--;
    return onLastRelease();
}

KVBlock::KVBlock(size_t size) : MemoryBlock(size) {}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
    return new KVBlock(*this);
}

EmbeddingWorkspace::EmbeddingWorkspace(size_t size) : MemoryBlock(size) {}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}


