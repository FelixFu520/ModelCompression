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
#include <vector>
#include <numeric> // for std::accumulate

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
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

double calculate_sum(const std::vector<double>& v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum;
}

double calculate_mean(const std::vector<double>& v) {
    double sum = calculate_sum(v);
    return sum / static_cast<float>(v.size());
}

double calculate_distance(const std::vector<double>& v) {
    double mean = calculate_mean(v);
    double max = 0;
    for (double i : v) {
        if (i > max) {
            max = i;
        }
    }
    return max - mean;
}

double calculate_fluctuation(const std::vector<double>& v) {
    double max = 0;
    for (double i : v) {
        if (i > max) {
            max = i;
        }
    }

    double min = 9999999;
    for (double i : v) {
        if (i < min) {
            min = i;
        }
    }
    return max - min;
}

double calculate_variance(const std::vector<double>& v) {
    double mean = calculate_mean(v);
    double variance = 0.0;
    for (auto x : v) {
        variance += pow(x - mean, 2);
    }
    variance /= static_cast<float>(v.size());
    return variance;
}

int main(int argc, char** argv){
    string engine_path, image_src, image_dst_label, image_dst_conf;
    int TYPE = 6; // 1,2->FP16; 3,4->INT8; 5,6->PTQ;
    if(TYPE==1){
        engine_path = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-train-fp16.trt";
        image_src = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14.bmp";
        image_dst_label = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14-trt_fp16-label.png";
        image_dst_conf = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14-trt_fp16-conf.png";
    }else if (TYPE==2){
        engine_path = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-train-fp16.trt";
        image_src = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21.bmp";
        image_dst_label = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21-trt_fp16-label.png";
        image_dst_conf = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21-trt_fp16-conf.png";
    }else if(TYPE==3){
        engine_path = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-train-int8.trt";
        image_src = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14.bmp";
        image_dst_label = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14-trt_int8-label.png";
        image_dst_conf = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14-trt_int8-conf.png";
    }else if(TYPE==4){
        engine_path = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-train-int8.trt";
        image_src = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21.bmp";
        image_dst_label = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21-trt_int8-label.png";
        image_dst_conf = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21-trt_int8-conf.png";
    }else if(TYPE==5){
        engine_path = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-ptq-calibrated-int8.trt";
        image_src = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14.bmp";
        image_dst_label = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14-ptq_int8-label.png";
        image_dst_conf = "/root/data/fufa/modelcompression/quantization/images/0000_Row006_Col036_00137_14-ptq_int8-conf.png";
    }else if(TYPE==6){
        engine_path = "/root/data/fufa/modelcompression/quantization/checkpoints/wafer-ptq-calibrated-int8.trt";
        image_src = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21.bmp";
        image_dst_label = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21-ptq_int8-label.png";
        image_dst_conf = "/root/data/fufa/modelcompression/quantization/images/0500_Row023_Col050_00953_21-ptq_int8-conf.png";
    }

    // 读图片
    Mat image = imread(image_src, 0);
    Mat fp32Mat;
    image.convertTo(fp32Mat, CV_32F);
    int H = 256;
    int W = 256;
    int N = 1;
    int C = 1;
    int Class_N = 2;
    int size_f = sizeof(float);

    // 初始化engine
    std::cout<< "init engine"<<std::endl;
    fstream file;
    file.open(engine_path, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    int64_t length = file.tellg();
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();
    // deserialize engine
    shared_ptr<NVLogger> m_clLogger = std::make_unique<NVLogger>(nvinfer1::ILogger::Severity::kINFO);
    shared_ptr<IRuntime> m_clRuntime = static_cast<shared_ptr<IRuntime>>(createInferRuntime(*m_clLogger));
    if(!m_clRuntime){ return 1;}
    shared_ptr<ICudaEngine> m_clEngine = static_cast<shared_ptr<ICudaEngine>>(m_clRuntime->deserializeCudaEngine(data.get(), length,nullptr));
    if(!m_clEngine){ return 1;}



    // 推理
    std::cout<<"save label&conf"<<std::endl;
    {
        // Context
        nvinfer1::IExecutionContext* ctx = m_clEngine->createExecutionContext();


        vector<void*> m_vBindings(2);
        vector<void*> m_vOutputs(1);
        cudaMalloc(&m_vBindings[0], N*C*H*W*size_f);
        cudaMalloc(&m_vBindings[1], N*Class_N*H*W*size_f);
        m_vOutputs[0] = malloc(N*Class_N*H*W*size_f);

        cudaMemcpy((float*)m_vBindings[0], fp32Mat.data, H*W*size_f, cudaMemcpyHostToDevice);

        ctx->executeV2(m_vBindings.data());

        cudaMemcpy((float*)m_vOutputs[0],  (float*)m_vBindings[1], N*Class_N*H*W*size_f, cudaMemcpyDeviceToHost);

        std::vector<cv::Mat> n_channels;  // 获取每类的概率图（包括多标签和单标签）
        for (int j = 0; j < Class_N; j++) {
            cv::Mat pred_c = cv::Mat(H, W, CV_32F, (float*)m_vOutputs[0] + j * H * W);
            cv::Mat pred_;
            pred_c.convertTo(pred_, CV_8U, 255);
            n_channels.push_back(pred_);
        }
        cv::Mat res;	// 存储合并后的图片
        cv::merge(n_channels, res);

        Mat label = Mat(H, W, CV_8U, Scalar::all(0));
        Mat conf = Mat(H, W, CV_8U, Scalar::all(0));

        for (int ii = 0; ii < H * W; ii++) {
            int h = ii / W;
            int w = ii % W;
            auto* p = (uchar*)res.ptr(h, w);				// 指向通道数组的指针
            auto p_max = std::max_element(p, p + Class_N);			// 获得通道数组中最大值指针
            auto labelValue = (uchar)std::distance(p, p_max);		// 求取最大值下表
            label.at<uchar>(h, w) = labelValue;						// label赋值
            if (labelValue != 0) {// conf赋值，如果label不为0则赋值，这个含义是将置信度图中背景类的值置为0
                conf.at<uchar>(h, w) = *p_max;
            }
        }

        cv::imwrite(image_dst_label, label);
        cv::imwrite(image_dst_conf, conf);

        for (auto & binding : m_vBindings) {
                cudaFree(binding);
        }
        for (auto & _output : m_vOutputs) {
            free(_output);
        }
        ctx->destroy();

    }


    // 统计时间
    std::cout<<"Time:"<<std::endl;
    vector<double> timesOfInfer;
    // Context
    nvinfer1::IExecutionContext* ctx = m_clEngine->createExecutionContext();
    vector<void*> m_vBindings(2);
    vector<void*> m_vOutputs(1);
    cudaMalloc(&m_vBindings[0], N*C*H*W*size_f);
    cudaMalloc(&m_vBindings[1], N*Class_N*H*W*size_f);
    m_vOutputs[0] = malloc(N*Class_N*H*W*size_f);

    for(int i=0;i<10000;i++){

        auto _start = std::chrono::high_resolution_clock::now(); // 记录开始时间
        cudaMemcpy((float*)m_vBindings[0], fp32Mat.data, H*W*size_f, cudaMemcpyHostToDevice);
        ctx->executeV2(m_vBindings.data());
        cudaMemcpy((float*)m_vOutputs[0],  (float*)m_vBindings[1], N*Class_N*H*W*size_f, cudaMemcpyDeviceToHost);
        auto _end = std::chrono::high_resolution_clock::now(); // 记录结束时间

        std::vector<cv::Mat> n_channels;  // 获取每类的概率图（包括多标签和单标签）
        for (int j = 0; j < Class_N; j++) {
            cv::Mat pred_c = cv::Mat(H, W, CV_32F, (float*)m_vOutputs[0] + j * H * W);
            cv::Mat pred_;
            pred_c.convertTo(pred_, CV_8U, 255);
            n_channels.push_back(pred_);
        }
        cv::Mat res;	// 存储合并后的图片
        cv::merge(n_channels, res);

        Mat label = Mat(H, W, CV_8U, Scalar::all(0));
        Mat conf = Mat(H, W, CV_8U, Scalar::all(0));

        for (int ii = 0; ii < H * W; ii++) {
            int h = ii / W;
            int w = ii % W;
            auto* p = (uchar*)res.ptr(h, w);				// 指向通道数组的指针
            auto p_max = std::max_element(p, p + Class_N);			// 获得通道数组中最大值指针
            auto labelValue = (uchar)std::distance(p, p_max);		// 求取最大值下表
            label.at<uchar>(h, w) = labelValue;						// label赋值
            if (labelValue != 0) {// conf赋值，如果label不为0则赋值，这个含义是将置信度图中背景类的值置为0
                conf.at<uchar>(h, w) = *p_max;
            }
        }

        cv::imwrite(image_dst_label, label);
        cv::imwrite(image_dst_conf, conf);



        auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start); // 计算时间间隔
        std::cout << "Infer Time is :" << std::to_string(_duration.count()) << " us" << std::endl;
        timesOfInfer.push_back(static_cast<double>(_duration.count()));    // 保存时间

    }
    for (auto & binding : m_vBindings) {
                cudaFree(binding);
    }
    for (auto & _output : m_vOutputs) {
            free(_output);
    }
    ctx->destroy();

    std::cout << "Total Time is :" << calculate_sum(timesOfInfer) << " us" << std::endl;
    std::cout << "Mean Time is :" << calculate_mean(timesOfInfer) << " us" << std::endl;
    std::cout << "Std Time is :" << calculate_variance(timesOfInfer) << " us" << std::endl;
    std::cout << "Volatility Time is :" << calculate_distance(timesOfInfer) << " us" << std::endl;
    std::cout << "Volatility Range Time is :" << calculate_fluctuation(timesOfInfer) << " us" << std::endl;
    /**
    ----------FP16
    Total Time is :6.83482e+06 us
    Mean Time is :683.482 us
    Std Time is :1707.03 us
    Volatility Time is :3455.52 us
    Volatility Range Time is :3479 us
    */
}