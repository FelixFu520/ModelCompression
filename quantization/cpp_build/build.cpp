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

class CNNCalibrator : public IInt8EntropyCalibrator2 {
    public:
        CNNCalibrator(const string &calibrationDataFolder, const string &imageSuffix,int calibrationNum) noexcept(false);
        ~CNNCalibrator() noexcept override;

    public:

        int32_t getBatchSize() const noexcept override ;
        bool getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept override;
        void const *readCalibrationCache(std::size_t &length) noexcept override;
        void writeCalibrationCache(void const *ptr, std::size_t length) noexcept override;

    private:
        vector<string> m_vImagesPath;               // 矫正数据

        int m_iCalibrationNum{0};                   // 矫正总次数
        int m_iCalibrationNumTimes{0};              // 现在是第几次矫正

        float* m_pBufferD{nullptr};                 // 输入在显存上的空间地址
        string m_sCacheFile{"cache.cache"};           // 缓存文件路径
    };

CNNCalibrator::CNNCalibrator(const string &calibrationDataFolder, const string &imageSuffix, int calibrationNum):
        m_iCalibrationNum(calibrationNum)
{
    // 获得所有图片路径
    cv::glob(calibrationDataFolder + "*" + imageSuffix, m_vImagesPath);

    // 分配显存
    cudaMalloc((void**)&m_pBufferD, sizeof(float) * 1 * 1 * 256 * 256);

}

bool CNNCalibrator::getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
    if (m_iCalibrationNumTimes < m_iCalibrationNum)
    {
        int idx = m_iCalibrationNumTimes%m_vImagesPath.size();
        Mat image = imread(m_vImagesPath[idx], 0);

        cv::Mat convertToFP32;
        image.convertTo(convertToFP32, CV_32F);
        cudaMemcpy(m_pBufferD, convertToFP32.data, 256*256* sizeof(float), cudaMemcpyHostToDevice);

        bindings[0] = m_pBufferD;
        m_iCalibrationNumTimes++;
        return true;
    }
    else
    {
        return false;
    }
}

CNNCalibrator::~CNNCalibrator() noexcept
{
    if (m_pBufferD != nullptr)
    {
        cudaFree(m_pBufferD);
    }
}

int32_t CNNCalibrator::getBatchSize() const noexcept
{
    return 1;
}

void const* CNNCalibrator::readCalibrationCache(std::size_t& length) noexcept
{

    std::fstream f;
    f.open(m_sCacheFile, std::fstream::in);
    if (f.fail())
    {
        return nullptr;
    }
    char* ptr = new char[length];
    if (f.is_open())
    {
        f >> ptr;
    }
    return ptr;
}

void CNNCalibrator::writeCalibrationCache(void const* ptr, std::size_t length) noexcept
{

    std::ofstream f(m_sCacheFile, std::ios::binary);
    if (f.fail())
    {
        return;
    }
    f.write(static_cast<char const*>(ptr), static_cast<long long>(length));
    if (f.fail())
    {
        return;
    }
    f.close();
}

int main(int argc, char** argv){
    string src = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-train.onnx";
    string dst, calibrator_dataset_path, imageSuffix;
    int calibrationNum;
    int TYPE = 1;
    int model = 1;
    if(TYPE==1){
        dst = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-train-fp16.trt";
        mode = 1;
    }
    else if(TYPE==2){
        dst = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-train-int8.trt";
        calibrator_dataset_path = "/root/data/fufa/modelcompression/Datasets/wafer/data/val/";
        imageSuffix = ".bmp";
        calibrationNum = 1000;
        mode = 2;
    }

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
    IInt8Calibrator* pCalibrator = nullptr;
    string cacheFile;
    if(mode == 1){
        config->setFlag(BuilderFlag::kFP16);
    }
    else if(mode == 2){
            config->setFlag(BuilderFlag::kINT8);
            pCalibrator = new CNNCalibrator(calibrator_dataset_path, imageSuffix, calibrationNum);
            config->setInt8Calibrator(pCalibrator);
            config->setDLACore(config->getDLACore());
    }

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