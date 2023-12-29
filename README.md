# Model Compression
## Quantization
### 数据说明
量化使用的是晶圆数据, 分析每类情况, 可以参考`tools/analysis_wafer.py`
```shell
label info of dataset is : {'0': 'background', '1': '凸起', '2': '凹坑', '3': '划伤', '4': '尘点'}
background has [5.2279805e+08] pixels.
凸起 has [3465.] pixels.
凹坑 has [16575.] pixels.
划伤 has [1661988.] pixels.
尘点 has [15197.] pixels.
```
由于类别间不均衡, 我把非背景全部归为一类了, 然后使用`tools/crop_images.py`裁剪图片, 为什么离线裁剪呢？
因为本实验只是为了验证量化时, 可以提速而且精度不会下降, 其余的不过度考虑, 就裁剪了, 没有做训练时在线裁剪
我是以`256*256`尺寸, `0.9*256`步长进行裁剪的, 共裁剪出8357张小图, 然后又将这些小图分为NG和BG两种类型, 
得到NG图1367张, BG图6990张.

然后, 从1367张图片中选择`1367*0.8=1093`张作为train, 其余的作为验证集, 然后从6990张图片中选取`1367*0.2*0.8`作为训练集, 选取
`1367*0.2*0.2`作为验证集.
```shell
BG images:6990
NG images:1367
训练集数量:1311, NG:1093, BG:218
验证集数量:328, NG:274, BG:54
```

### 环境
```shell
sudo docker build -t quantization:20231225 . -f DockerFileQuantization
sudo docker run -p 10023:22 --name fufa_quant -itd -v /data:/root/data --gpus all --privileged --shm-size=64g quantization:20231225
```

### 程序说明
```shell
一、训练和验证
python train.py
python train_predict.py # 注意修改下pth的路径

二、对导出onnx, 转成trt, 使用FP16, INT8(TensorRT API 量化)推理
python train_convert_onnx.py # 模型转换
然后cmake、make，执行推理, 可能会修改一些C++代码
cd cpp_build && mkdir build && cd build
cmake ..
make
./onnx2trt
cd cpp_infer && mkdir build && cd build
cmake ..
make
./infer

三、 使用pytorch-quantization库量化
python ptq.py
python ptq_predict.py
python ptq_convert_onnx.py
转trt, 然后测试

python qat.py
python qat_predict.py
python qat_conver_onnx.py
转trt, 然后测试

```
### 测试结果
1. TensorRT FP16: 685us
2. TensorRT INT8: 572us
3. pytorch-quantization+TensorRT INT8(PTQ): 660us
4. pytorch-quantization+TensorRT INT8(QAT): 669us
