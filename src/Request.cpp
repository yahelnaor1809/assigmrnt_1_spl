#include "Request.h"
#include "Model.h" 
#include "GPU.h"

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), tokens(), memoryBlocks(), reservedBlocks(0) {
    tokens = model.tokenizer(prompt);
}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, 
                                     std::string prompt, std::string scriptedOutput, std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens), outputTokens(), currentTokenIndex(0) {
    
    std::vector<std::string> allOutputTokens = model.tokenizer(scriptedOutput);
    
    for(std::size_t i = 0; i < allOutputTokens.size() && i < maxOutputTokens; i++) {
        outputTokens.push_back(allOutputTokens[i]);
    }
}
std::size_t GenerativeRequest::getMaxNeededMemory() const {
    return model.getEmbeddingSize() * (tokens.size() + maxOutputTokens);
}

std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}

void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt), currentTokenIndex(0), outputResult("") {
}
bool EmbeddingRequest::finished() const {
    return currentTokenIndex >= tokens.size();
}

std::size_t EmbeddingRequest::getNeedToUnreserve() const {
    if (reservedBlocks > memoryBlocks.size()) {
        return reservedBlocks - memoryBlocks.size();
    }
    return 0;
}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    reservedBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    gpu.reserve(reservedBlocks, *this);
    
    for (std::size_t i = 0; i < reservedBlocks; i++) {
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot); 
    }
}
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens);
    std::size_t h = match.matchedTokens;
    
    std::size_t n = tokens.size();
    std::size_t m = maxOutputTokens;
    
    return model.getEmbeddingSize() * (n - h + m);
}

void GenerativeRequest::reserveMemory(GPU& gpu) {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens);
    
    for (std::size_t slot : match.blockIds) {
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
    }
    
    reservedBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    
    gpu.reserve(reservedBlocks, *this);
}
Request* GenerativeRequest::clone(const Model& clonedModel) const {
    GenerativeRequest* copy = new GenerativeRequest(this->id, clonedModel, this->admissionCycle, "", "", this->maxOutputTokens);
    copy->tokens = this->tokens;
    copy->scriptedOutput = this->scriptedOutput;
    copy->outputTokens = this->outputTokens;
    copy->memoryBlocks = this->memoryBlocks;
    copy->reservedBlocks = this->reservedBlocks;
    copy->currentTokenIndex = this->currentTokenIndex;
    return copy;
}

std::size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

bool GenerativeRequest::finished() const {
    return currentTokenIndex >= tokens.size() + outputTokens.size();
}

std::string GenerativeRequest::getOutput() const {
    std::string out = "";
    for (std::size_t i = 0; i < outputTokens.size(); i++) {
        out += outputTokens[i];
        if (i < outputTokens.size() - 1) {
            out += " ";
        }
    }
    return out;
}
void GenerativeRequest::act(GPU& gpu) {
    std::size_t totalTokens = tokens.size() + outputTokens.size();
    if (currentTokenIndex >= totalTokens) return;
    
    std::size_t t = gpu.getBlockSize() / model.getEmbeddingSize();
    
    if (currentTokenIndex == 0 && !memoryBlocks.empty()) {
        currentTokenIndex = memoryBlocks.size() * t;
        if (currentTokenIndex >= totalTokens) return;
    }

    std::vector<std::string> prefix;
    for (std::size_t i = 0; i <= currentTokenIndex; i++) {
        if (i < tokens.size()) {
            prefix.push_back(tokens[i]); 
        } else {
            prefix.push_back(outputTokens[i - tokens.size()]); 
        }
    }

    std::size_t blockIndex = currentTokenIndex / t;
    std::size_t offset = (currentTokenIndex % t) * model.getEmbeddingSize();

    if (blockIndex >= memoryBlocks.size()) {
        std::size_t newSlot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(newSlot);
        if (reservedBlocks > 0) reservedBlocks--; 
    }

    std::size_t slot = memoryBlocks[blockIndex];
    model.generateTokenEmbedding(prefix, gpu, slot, offset);

    currentTokenIndex++;

    if (currentTokenIndex % t == 0) {
        gpu.assignPrefixToBlock(slot, prefix, model.getId());
    }
}

Request* EmbeddingRequest::clone(const Model& clonedModel) const {
    EmbeddingRequest* copy = new EmbeddingRequest(this->id, clonedModel, this->admissionCycle, "");
    
    copy->tokens = this->tokens;
    copy->memoryBlocks = this->memoryBlocks;
    copy->reservedBlocks = this->reservedBlocks;
    copy->currentTokenIndex = this->currentTokenIndex;
    copy->outputResult = this->outputResult;
    
    return copy;
}

void EmbeddingRequest::act(GPU& gpu) {
    if (currentTokenIndex >= tokens.size()) return;

    std::size_t e = model.getEmbeddingSize();
    
    std::size_t sum_slot = memoryBlocks[0];
    std::size_t sum_offset = 0; 
    
    std::size_t token_slot = (memoryBlocks.size() > 1) ? memoryBlocks[1] : memoryBlocks[0];
    std::size_t token_offset = (memoryBlocks.size() > 1) ? 0 : e;

    std::vector<std::string> singleToken = { tokens[currentTokenIndex] };
    model.generateTokenEmbedding(singleToken, gpu, token_slot, token_offset);

    for (std::size_t i = 0; i < e; i++) {
        int current_sum = gpu.readFromBlock(sum_slot, sum_offset + i);
        int token_val = gpu.readFromBlock(token_slot, token_offset + i);
        gpu.writeToBlock(sum_slot, sum_offset + i, current_sum + token_val);
    }

    currentTokenIndex++; 

    if (currentTokenIndex == tokens.size()) {
        outputResult = "[";
        for (std::size_t i = 0; i < e; i++) {
            int final_sum = gpu.readFromBlock(sum_slot, sum_offset + i);
            int mean_val = final_sum / static_cast<int>(tokens.size()); 
            outputResult += std::to_string(mean_val);
            
            if (i < e - 1) {
                outputResult += ", ";
            }
        }
        outputResult += "]";
    }
}

std::string EmbeddingRequest::getOutput() const {
    return outputResult;
}
void Request::setAssignedGpuIndex(int index) {
    assignedGpuIndex = index;
}

int Request::getAssignedGpuIndex() const {
    return assignedGpuIndex;
}

const std::vector<std::size_t>& Request::getMemoryBlocks() const {
    return memoryBlocks;
}
std::size_t EmbeddingRequest::getMaxNeededMemory() const {
    return 2 * model.getEmbeddingSize();
}

std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const {
    return 2 * model.getEmbeddingSize();
}

std::vector<std::string> EmbeddingRequest::getCacheableTokens() const {
    return std::vector<std::string>();
}

void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

void Request::deallocateMemory(GPU& gpu) {
    gpu.unreserve(getNeedToUnreserve(), *this);
    for (std::size_t slot : memoryBlocks) {
        gpu.release(slot);
    }
    memoryBlocks.clear();
    reservedBlocks = 0;
}