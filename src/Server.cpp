#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"
#include "GPU.h"
#include "Request.h"
#include <iostream>

#include "Model.h"
#include "SchedulingPolicy.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }

    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}
Server::~Server() {
    for(Model* m : models){
            delete m;
    }
    for(Request* r : nonSubmittedRequests){
        delete r;
    }
    for(Request* r: waitingRequests){
        delete r;
    }
    for(Request* r: activeRequests){
        delete r;
    }

    delete policy;
    policy = nullptr;
}
Server::Server(const Server& other) : gpus(other.gpus), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr), currentCycle(0) {        
    for(Model* m: other.models){
            this->models.push_back(new Model(*m));
        }

        this->gpus = other.gpus;
        this->policy = other.policy->clone();

        for(Request* r: other.nonSubmittedRequests){
            for(Model* m: this->models){
                if(r->getModel().getId() == m->getId()){
                    this->nonSubmittedRequests.push_back(r->clone(*m));
                    break;
                }
            }
        }

        for(Request* r: other.waitingRequests){
            for(Model* m: this->models){
                if(r->getModel().getId() == m->getId()){
                    this->waitingRequests.push_back(r->clone(*m));
                    break;
                }
            }
        }
        for(Request* r: other.activeRequests){
            for(Model* m: this->models){
                if(r->getModel().getId() == m->getId()){
                    this->activeRequests.push_back(r->clone(*m));
                    break;
                }
            }
        }
    }

    Server& Server::operator=(const Server& other) {
        if(this == &other){
            return *this;
        }
        
        this->currentCycle = other.currentCycle;

        for(Model* m : models){
                delete m;
        }
        models.clear();

        for(Request* r : nonSubmittedRequests){
            delete r;
        }
        nonSubmittedRequests.clear();

        for(Request* r: waitingRequests){
            delete r;
        }
        waitingRequests.clear();

        for(Request* r: activeRequests){
            delete r;
        }
        activeRequests.clear();

        delete policy;
        policy = nullptr;

        for(Model* m: other.models){
            this->models.push_back(new Model(*m));
        }

        this->gpus = other.gpus;
        this->policy = other.policy->clone();

        for(Request* r: other.nonSubmittedRequests){
            for(Model* m: this->models){
                if(r->getModel().getId() == m->getId()){
                    this->nonSubmittedRequests.push_back(r->clone(*m));
                    break;
                }
            }
        }

        for(Request* r: other.waitingRequests){
            for(Model* m: this->models){
                if(r->getModel().getId() == m->getId()){
                    this->waitingRequests.push_back(r->clone(*m));
                    break;
                }
            }
        }
        for(Request* r: other.activeRequests){
            for(Model* m: this->models){
                if(r->getModel().getId() == m->getId()){
                    this->activeRequests.push_back(r->clone(*m));
                    break;
                }
            }
        }
        return *this;
    }
Server::Server(Server&& other) : gpus(), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr), currentCycle(other.currentCycle) {    this->activeRequests = other.activeRequests;
    this->nonSubmittedRequests = other.nonSubmittedRequests;
    this->waitingRequests = other.waitingRequests;
    this->policy = other.policy;
    this->gpus = other.gpus;
    this->models = other.models;

    other.models.clear();
    other.activeRequests.clear();
    other.nonSubmittedRequests.clear();
    other.waitingRequests.clear();
    other.policy = nullptr;
}

Server& Server::operator=(Server&& other){
    if(this == &other){
            return *this;
    }

    for(Model* m : models){
            delete m;
    }
    models.clear();

    for(Request* r : nonSubmittedRequests){
        delete r;
    }
    nonSubmittedRequests.clear();

    for(Request* r: waitingRequests){
        delete r;
    }
    waitingRequests.clear();

    for(Request* r: activeRequests){
        delete r;
    }
    activeRequests.clear();

    delete policy;
    policy = nullptr;

    this->activeRequests = other.activeRequests;
    this->nonSubmittedRequests = other.nonSubmittedRequests;
    this->waitingRequests = other.waitingRequests;
    this->policy = other.policy;
    this->gpus = other.gpus;
    this->models = other.models;
    this->currentCycle = other.currentCycle;

    other.models.clear();
    other.activeRequests.clear();
    other.nonSubmittedRequests.clear();
    other.waitingRequests.clear();
    other.policy = nullptr;

    return *this;
}

Server::Server(std::string configPath) : gpus(), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr), currentCycle(0) {    
    Config config = readConfig(configPath);
    for(GPUConfig g: config.gpus){
        this->gpus.push_back(GPU(g.memoryBytes / config.blockSize,config.blockSize));
    }
    
    for(ModelConfig m: config.models){
        this->models.push_back(new Model(m.id, m.embeddingSize));
    }
  
    if(config.policy == "arrivalOrder"){
        this->policy = new ArrivalOrderPolicy();
    } else if(config.policy == "shortestJob"){
        this->policy = new ShortestJobPolicy();
    } else if(config.policy == "longestPrefix"){
        this->policy = new LongestPrefixPolicy();
    }
    for(RequestConfig r: config.requests){
        Model* matchedModel = nullptr;
        for(Model* m : this->models){
            if(m->getId() == r.modelId){
                matchedModel = m;
                break;
            }
        }

        if(r.type == RequestType::Generative){
            this->nonSubmittedRequests.push_back(
                new GenerativeRequest(r.id, *matchedModel, r.admissionCycle, r.prompt, r.scriptedOutput, r.maxOutputTokens)
            );
        } else { // TextEmbedder
            this->nonSubmittedRequests.push_back(
                new EmbeddingRequest(r.id, *matchedModel, r.admissionCycle, r.prompt)
            );
        }
    }

}
void Server::serve() {
    while(step()){
    }
    int totalCacheHits = 0;
    int totalEvictions = 0;

    for(GPU g: gpus){
        totalCacheHits += g.getCacheHits();
        totalEvictions += g.getEvictions();
    } 
    std::cout << "summary: cycles=" <<  currentCycle
            << " cache_hits=" << totalCacheHits 
            << " evictions=" << totalEvictions << std::endl;
    }

bool Server::step() {
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty()) {
        return false;
    }

    for (size_t i = 0; i < nonSubmittedRequests.size();) {
        if (nonSubmittedRequests[i]->getAdmissionCycle() == currentCycle) {
            Request* req = nonSubmittedRequests[i];
            
            bool canFit = false;
            for (const GPU& gpu : gpus) {
                if (gpu.getBlocksForMemory(req->getMaxNeededMemory()) <= gpu.getBlocksNum()) {
                    canFit = true;
                    break;
                }
            }
            
            if (!canFit) {
                std::cout << "rejected " << req->getId() << " cycle=" << currentCycle << std::endl;
                delete req;
            } else {
                waitingRequests.push_back(req);
            }
            nonSubmittedRequests.erase(nonSubmittedRequests.begin() + i);
        } else {
            i++; 
        }
    }

    std::vector<const Request*> eligible;
    for (size_t i = 0; i < waitingRequests.size(); i++) {
        eligible.push_back(waitingRequests[i]);
    }
    std::vector<int> order = policy->choose(eligible, gpus);
    std::vector<bool> admitted(waitingRequests.size(), false);

    for (int idx : order) {
        Request* req = waitingRequests[idx];
        
        int bestGpuIndex = -1;
        std::size_t maxCacheInBestGpu = 0;
        int bestCacheGpuIndex = -1; 
        std::size_t absoluteMaxCache = 0;
        std::vector<std::size_t> blocksToCopy;

        for (std::size_t i = 0; i < gpus.size(); i++) {
            std::vector<std::string> cacheableTokens = req->getCacheableTokens();
            PrefixMatch match;
            if (!cacheableTokens.empty()) {
                match = gpus[i].findLongestCachedPrefix(req->getModel().getId(), req->getModel().getEmbeddingSize(), cacheableTokens);
            } else {
                match.matchedTokens = 0;
            }

            if (match.matchedTokens > absoluteMaxCache) {
                absoluteMaxCache = match.matchedTokens;
                bestCacheGpuIndex = i;
                blocksToCopy = match.blockIds;
            }

            std::size_t blocksNeeded = gpus[i].getBlocksForMemory(req->getNeededMemory(gpus[i]));
            
            if (gpus[i].availableSlots() >= blocksNeeded) {
                if (bestGpuIndex == -1 || match.matchedTokens > maxCacheInBestGpu) {
                    bestGpuIndex = i;
                    maxCacheInBestGpu = match.matchedTokens;
                }
            }
        }

        if (bestGpuIndex != -1) {
            if (bestCacheGpuIndex != -1 && bestCacheGpuIndex != bestGpuIndex && absoluteMaxCache > maxCacheInBestGpu) {
                gpus[bestGpuIndex].copyMemoryFromOtherGPU(gpus[bestCacheGpuIndex], blocksToCopy);
            }

            req->reserveMemory(gpus[bestGpuIndex]);
            activeRequests.push_back(req);
            req->setAssignedGpuIndex(bestGpuIndex);
            admitted[idx] = true;
        }
    }

    std::vector<Request*> stillWaiting;
    for (size_t i = 0; i < waitingRequests.size(); i++) {
        if (!admitted[i]) {
            stillWaiting.push_back(waitingRequests[i]);
        }
    }
    waitingRequests = stillWaiting;

    auto activeIt = activeRequests.begin();
    while (activeIt != activeRequests.end()) {
        Request* req = *activeIt;
        int assignedGpuIndex = req->getAssignedGpuIndex();
        
        req->act(gpus[assignedGpuIndex]); 

        if (req->finished()) {
            req->deallocateMemory(gpus[assignedGpuIndex]);
            
            std::cout << "finished " << req->getId() << " cycle=" << currentCycle 
                      << " gpu=" << assignedGpuIndex << ": " << req->getOutput() << std::endl;
            
            delete req;
            activeIt = activeRequests.erase(activeIt);
        } else {
            ++activeIt;
        }
    }

    currentCycle++;
    return true;
}