#include <cstdio>
#include <dlpack/dlpack.h>
// #include <opencv4/opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>

using namespace std;
using namespace tvm;

void Mat_to_CHW(float *data, cv::Mat &frame){
    assert(data && !frame.empty());
    unsigned int volChl = 640 * 640;

    for(int c = 0; c < 3; ++c)
    {
        for (unsigned j = 0; j < volChl; ++j)
            data[c*volChl + j] = static_cast<float>(float(frame.data[j * 3 + c]) / 255.0);
    }

}

int main(){
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("tvm_output_lib/yolov5s.so");

    // std::ifstream json_in;
    // json_in.open("./tvm_output_lib/yolov5m.json", std::ios::in);
    std::ifstream json_in("tvm_output_lib/yolov5s.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // parameters in binary
    std::ifstream params_in("tvm_output_lib/yolov5s.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

    // tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_dylib, device_type, device_id);
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data, mod_dylib, device_type, device_id);
    // printf("1.51.51.51.5\n");
    DLTensor *x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 640, 640};
    
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    
    // 这里依然读取了papar.png这张图
    cv::Mat image = cv::imread("./bus.jpg");
    cv::Mat frame, input;
    cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, input,  cv::Size(640,640));

    float data[640 * 640 * 3];
    // 在这个函数中 将OpenCV中的图像数据转化为CHW的形式 
    Mat_to_CHW(data, input);
    
    
    // x为之前的张量类型 data为之前开辟的浮点型空间
    memcpy(x->data, &data, 3 * 640 * 640 * sizeof(float));

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("images", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);
    
    DLTensor* y;
    // int out_ndim = 2;
    int out_ndim = 3;
    // int64_t out_shape[2] = {1, 3,};
    int64_t out_shape[3] = {1, 25200, 85};
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

    run();
    get_output(0, y);

    // 将输出的信息打印出来
    auto result = static_cast<float*>(y->data);
    for (int i = 0; i < 3; i++)
        cout<<result[i]<<endl;
    }

