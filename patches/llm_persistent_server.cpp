/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "profileFormatter.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/streaming.h"

#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

constexpr char const* kReadyPrefix = "EDGELLM_SERVER_READY ";
constexpr char const* kResponsePrefix = "EDGELLM_SERVER_RESPONSE ";
constexpr char const* kErrorPrefix = "EDGELLM_SERVER_ERROR ";

enum PersistentServerOptionId : int
{
    HELP = 900,
    ENGINE_DIR = 901,
    MULTIMODAL_ENGINE_DIR = 902,
    BATCH_SIZE = 903,
    MAX_GENERATE_LENGTH = 904,
};

struct PersistentServerArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir;
    int32_t batchSize{-1};
    int64_t maxGenerateLength{-1};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " --engineDir=<path> [--multimodalEngineDir=<path>] [--batchSize=<n>]"
                 " [--maxGenerateLength=<n>]\n\n"
              << "Reads newline-delimited JSON commands on stdin and emits protocol lines on stdout.\n"
              << "Request command schema:\n"
              << "  {\"id\":\"optional\", \"input_file\":\"/path/request.json\","
                 " \"batch_size\":1, \"max_generate_length\":48}\n"
              << "Shutdown command schema:\n"
              << "  {\"shutdown\":true}\n";
}

bool parseArgs(PersistentServerArgs& args, int argc, char* argv[])
{
    static struct option options[] = {{"help", no_argument, 0, PersistentServerOptionId::HELP},
        {"engineDir", required_argument, 0, PersistentServerOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, PersistentServerOptionId::MULTIMODAL_ENGINE_DIR},
        {"batchSize", required_argument, 0, PersistentServerOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, PersistentServerOptionId::MAX_GENERATE_LENGTH},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        try
        {
            switch (opt)
            {
            case PersistentServerOptionId::HELP: args.help = true; return true;
            case PersistentServerOptionId::ENGINE_DIR: args.engineDir = optarg; break;
            case PersistentServerOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
            case PersistentServerOptionId::BATCH_SIZE:
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    std::cerr << "Invalid --batchSize: " << optarg << "\n";
                    return false;
                }
                break;
            case PersistentServerOptionId::MAX_GENERATE_LENGTH:
                args.maxGenerateLength = std::stoll(optarg);
                if (args.maxGenerateLength <= 0)
                {
                    std::cerr << "Invalid --maxGenerateLength: " << optarg << "\n";
                    return false;
                }
                break;
            default: return false;
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "Invalid option value: " << e.what() << "\n";
            return false;
        }
    }

    if (args.help)
    {
        return true;
    }
    if (args.engineDir.empty())
    {
        std::cerr << "--engineDir is required\n";
        return false;
    }
    return true;
}

void emitProtocolLine(char const* prefix, Json const& payload)
{
    std::cout << prefix << payload.dump() << std::endl;
}

Json failurePayload(std::string const& id, std::string const& error)
{
    Json payload;
    payload["ok"] = false;
    payload["id"] = id;
    payload["error"] = error;
    return payload;
}

std::vector<int32_t> parseScoreTokenIds(Json const& command)
{
    std::vector<int32_t> tokenIds;
    if (!command.contains("score_token_ids"))
    {
        return tokenIds;
    }
    if (!command["score_token_ids"].is_array())
    {
        throw std::runtime_error("score_token_ids must be an array of integer token IDs");
    }
    for (auto const& item : command["score_token_ids"])
    {
        if (!item.is_number_integer())
        {
            throw std::runtime_error("score_token_ids must contain only integer token IDs");
        }
        tokenIds.push_back(item.get<int32_t>());
    }
    return tokenIds;
}

double sigmoid(double x)
{
    if (x >= 0.0)
    {
        double const z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    double const z = std::exp(x);
    return z / (1.0 + z);
}

void appendTokenLogits(Json& responseJson, rt::LLMInferenceRuntime const& runtime, size_t batchIdx,
    std::vector<int32_t> const& tokenIds)
{
    if (tokenIds.empty())
    {
        return;
    }
    rt::Tensor const* logitsTensor = runtime.getBaseModelOutputLogits();
    if (logitsTensor == nullptr || logitsTensor->isEmpty())
    {
        throw std::runtime_error("output logits are unavailable after handleRequest()");
    }
    auto const& shape = logitsTensor->getShape();
    if (shape.getNumDims() != 2)
    {
        throw std::runtime_error("output logits tensor must have shape [batch, vocab]");
    }
    int64_t const batchSize = shape[0];
    int64_t const vocabSize = shape[1];
    if (static_cast<int64_t>(batchIdx) >= batchSize)
    {
        throw std::runtime_error("requested batch index is outside output logits batch dimension");
    }

    float const* logits = logitsTensor->dataPointer<float>();
    Json tokenLogits = Json::array();
    std::vector<float> hostLogits;
    hostLogits.reserve(tokenIds.size());
    for (int32_t tokenId : tokenIds)
    {
        if (tokenId < 0 || static_cast<int64_t>(tokenId) >= vocabSize)
        {
            throw std::runtime_error("score token ID is outside output vocabulary");
        }
        float value = 0.0F;
        cudaError_t err = cudaMemcpy(&value, logits + static_cast<int64_t>(batchIdx) * vocabSize + tokenId,
            sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("failed to copy output logit: ") + cudaGetErrorString(err));
        }
        hostLogits.push_back(value);
        tokenLogits.push_back(value);
    }
    responseJson["score_token_ids"] = tokenIds;
    responseJson["token_logits"] = tokenLogits;
    if (hostLogits.size() >= 2)
    {
        responseJson["token_logit_score"] = sigmoid(static_cast<double>(hostLogits[0] - hostLogits[1]));
    }
}

Json runInputFile(rt::LLMInferenceRuntime& runtime, cudaStream_t stream, Json const& command,
    PersistentServerArgs const& defaultArgs)
{
    std::string const id = command.value("id", "");
    if (!command.contains("input_file") || !command["input_file"].is_string())
    {
        return failurePayload(id, "command requires string field: input_file");
    }

    std::filesystem::path const inputFile = command["input_file"].get<std::string>();
    int32_t const batchSize
        = command.contains("batch_size") ? command["batch_size"].get<int32_t>() : defaultArgs.batchSize;
    int64_t const maxGenerateLength = command.contains("max_generate_length")
        ? command["max_generate_length"].get<int64_t>()
        : defaultArgs.maxGenerateLength;
    std::vector<int32_t> scoreTokenIds;
    try
    {
        scoreTokenIds = parseScoreTokenIds(command);
    }
    catch (std::exception const& e)
    {
        return failurePayload(id, e.what());
    }

    auto const start = std::chrono::steady_clock::now();

    std::unordered_map<std::string, std::string> requestLoraWeightsMap;
    std::vector<rt::LLMGenerationRequest> batchedRequests;
    try
    {
        std::tie(requestLoraWeightsMap, batchedRequests)
            = exampleUtils::parseRequestFile(inputFile, batchSize, maxGenerateLength);
    }
    catch (std::exception const& e)
    {
        return failurePayload(id, std::string("failed to parse input file: ") + e.what());
    }

    if (!requestLoraWeightsMap.empty())
    {
        return failurePayload(id, "LoRA weights are not supported by llm_persistent_server after startup");
    }
    if (batchedRequests.empty())
    {
        return failurePayload(id, "input file did not contain any valid batched requests");
    }

    Json payload;
    payload["ok"] = true;
    payload["id"] = id;
    payload["responses"] = Json::array();
    payload["batched_request_count"] = batchedRequests.size();

    bool allOk = true;
    std::string const errorText = "TensorRT Edge LLM cannot handle this request. Fails.";

    for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
    {
        auto& request = batchedRequests[requestIdx];
        rt::LLMGenerationResponse response;
        bool requestStatus = false;
        std::string requestError;
        try
        {
            requestStatus = runtime.handleRequest(request, response, stream);
        }
        catch (std::exception const& e)
        {
            requestStatus = false;
            requestError = e.what();
        }

        if (!requestStatus)
        {
            allOk = false;
        }

        for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
        {
            Json responseJson;
            bool const hasOutputText = requestStatus && batchIdx < response.outputTexts.size();
            std::string const outputText = hasOutputText ? response.outputTexts[batchIdx] : errorText;
            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            responseJson["request_idx"] = requestIdx;
            responseJson["batch_idx"] = batchIdx;
            responseJson["finish_reason"] = (requestStatus && batchIdx < response.finishReasons.size())
                ? rt::finishReasonName(response.finishReasons[batchIdx])
                : "error";
            if (!requestError.empty())
            {
                responseJson["error"] = requestError;
            }
            if (requestStatus && !scoreTokenIds.empty())
            {
                try
                {
                    appendTokenLogits(responseJson, runtime, batchIdx, scoreTokenIds);
                }
                catch (std::exception const& e)
                {
                    allOk = false;
                    responseJson["finish_reason"] = "error";
                    responseJson["error"] = e.what();
                }
            }
            payload["responses"].push_back(responseJson);
        }
    }

    auto const end = std::chrono::steady_clock::now();
    payload["ok"] = allOk;
    payload["latency_ms"]
        = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    return payload;
}

} // namespace

int main(int argc, char* argv[])
{
    PersistentServerArgs args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    auto pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    try
    {
        std::unordered_map<std::string, std::string> loraWeightsMap;
        runtime = std::make_unique<rt::LLMInferenceRuntime>(
            args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    }
    catch (std::exception const& e)
    {
        Json payload;
        payload["ok"] = false;
        payload["error"] = std::string("failed to initialize runtime: ") + e.what();
        emitProtocolLine(kErrorPrefix, payload);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    Json ready;
    ready["ok"] = true;
    ready["engine_dir"] = args.engineDir;
    ready["multimodal_engine_dir"] = args.multimodalEngineDir;
    ready["cuda_graph_captured"] = false;
    emitProtocolLine(kReadyPrefix, ready);

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }

        Json command;
        try
        {
            command = Json::parse(line);
        }
        catch (std::exception const& e)
        {
            emitProtocolLine(kResponsePrefix, failurePayload("", std::string("invalid command JSON: ") + e.what()));
            continue;
        }

        std::string const id = command.value("id", "");
        if (command.value("shutdown", false))
        {
            Json payload;
            payload["ok"] = true;
            payload["id"] = id;
            payload["shutdown"] = true;
            emitProtocolLine(kResponsePrefix, payload);
            break;
        }

        try
        {
            emitProtocolLine(kResponsePrefix, runInputFile(*runtime, stream, command, args));
        }
        catch (std::exception const& e)
        {
            emitProtocolLine(kResponsePrefix, failurePayload(id, std::string("request failed: ") + e.what()));
        }
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
