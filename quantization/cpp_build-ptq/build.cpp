/**************************************************
* @author: FuFa
* @email: alanmathisonturing@163.com
* @date: 2023/12/28 9:34
* @description:
***************************************************/
#include <iostream>
#include <fstream>
#include <memory>
#include <thread>
#include <string>

#include <opencv2/opencv.hpp>
#include <NvInferVersion.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>


using namespace std;
using namespace nvinfer1;
using namespace cv;

class NVLogger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit NVLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO):
            reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
                    std::cout<<msg<<std::endl;
    }
};

int main(int argc, char** argv){
    string src = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-ptq-calibrated.onnx";
    string dst = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-ptq-calibrated-int8.trt";

    // builder
    auto gLogger = std::make_unique<NVLogger>(nvinfer1::ILogger::Severity::kINFO);
    auto builder = unique_ptr<IBuilder>(createInferBuilder(*gLogger));
    if(!builder){ return 1;}

    // network
    const auto explicitBatch = 1U << static_cast<int32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network){ return 1;}

    // parser
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *gLogger));
    if(!parser){ return 1;}

    // parsed
    auto verbosity = static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE);
    auto parsed = parser->parseFromFile(src.c_str(), verbosity);
    if(!parsed){ return 1;}

    // config
    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if(!config){ return 1;}

    // select mode
    config->setFlag(BuilderFlag::kINT8);

    // plan
    auto plan = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if(!plan){ return 1;}

    // runtime
    auto runtime = unique_ptr<IRuntime>(nvinfer1::createInferRuntime(*gLogger));
    if(!runtime){ return 1;}

    // engine
    auto engine = unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if(!engine){ return 1;}

    // save
    auto serializedModel = unique_ptr<IHostMemory>(engine->serialize());
    std::ofstream p(dst.c_str(), std::ios::binary);
    p.write((const char*)serializedModel->data(), (signed long long)serializedModel->size());
    p.close();
}