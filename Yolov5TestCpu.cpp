#include <cstdio>
#include <dlpack/dlpack.h>
// #include <opencv4/opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <vector>
#include <string>
#include "time.h"

using namespace std;
using namespace tvm;

 
std::vector<std::string> classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

void Mat_to_CHW(float *data, cv::Mat &frame){
    assert(data && !frame.empty());
    unsigned int volChl = 640 * 640;

    for(int c = 0; c < 3; ++c)
    {
        for (unsigned j = 0; j < volChl; ++j)
            data[c*volChl + j] = static_cast<float>(float(frame.data[j * 3 + c]) / 255.0);
    }

}

// float xywh2xyxy(float x, float y, float w, float h){
//     // Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
//     float x1 = x - w / 2;  // top left x
//     float y1 = y - h / 2;  // top left y
//     float x2 = x + w / 2;  // bottom right x
//     float y2 = y + h / 2;  // bottom right y
//     return {x1, y1, x2, y2};
// }

void xywh2lrwh(float x, float y, float w, float h, float &left, float &right){
    // Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    left = x - w / 2;  // top left x
    right = y - h / 2;  // top left y
}

// void nms(vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname = "Union"){
 
//     if(boundingBox_.empty()){
//         return;
//     }
//     //对各个候选框根据score的大小进行升序排列
//     sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
//     float IOU = 0;
//     float maxX = 0;
//     float maxY = 0;
//     float minX = 0;
//     float minY = 0;
//     vector<int> vPick;
//     int nPick = 0;
//     multimap<float, int> vScores;   //存放升序排列后的score和对应的序号
//     const int num_boxes = boundingBox_.size();
// 	vPick.resize(num_boxes);
// 	for (int i = 0; i < num_boxes; ++i){
// 		vScores.insert(pair<float, int>(boundingBox_[i].score, i));
// 	}
//     while(vScores.size() > 0){
//         int last = vScores.rbegin()->second;  //反向迭代器，获得vScores序列的最后那个序列号
//         vPick[nPick] = last;
//         nPick += 1;
//         for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
//             int it_idx = it->second;
//             maxX = max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
//             maxY = max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
//             minX = min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
//             minY = min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
//             //转换成了两个边界框相交区域的边长
//             maxX = ((minX-maxX+1)>0)? (minX-maxX+1) : 0;
//             maxY = ((minY-maxY+1)>0)? (minY-maxY+1) : 0;
//             //求交并比IOU
            
//             IOU = (maxX * maxY)/(boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
//             if(IOU > overlap_threshold){
//                 it = vScores.erase(it);    //删除交并比大于阈值的候选框,erase返回删除元素的下一个元素
//             }else{
//                 it++;
//             }
//         }
//     }
    
//     vPick.resize(nPick);
//     vector<Bbox> tmp_;
//     tmp_.resize(nPick);
//     for(int i = 0; i < nPick; i++){
//         tmp_[i] = boundingBox_[vPick[i]];
//     }
//     boundingBox_ = tmp_;
// }

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
 
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    else
    {
        std::cout<<"classes is empty..."<<std::endl;
    }
 
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
    // cout << label << endl;
    // cv::imwrite("C_Res.jpg", frame);
}

void non_max_suppression(cv::Mat frame, int64_t out_shape[], vector<vector<float>> output, float conf_thres, float iou_thres){
    // number of classes
    int nc = out_shape[3] - 5;
    // maximum number of boxes into nms()
    int max_nms = 30000;
    // int xc = out_shape[3] > conf_thres  # candidates
    vector<vector<float>> filterout;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float o_w = (float)frame.cols;
    float o_h = (float)frame.rows;
    float r_w = o_w / 640;
    float r_h = o_h / 640;

    for (auto oneobj = output.begin(); oneobj != output.end(); oneobj++){
        // cout << "conf is : " << (*oneobj).at(4) << endl;
        if((*oneobj).at(4) > conf_thres){
            float x = (*oneobj).at(0);
            float y = (*oneobj).at(1);
            float w = (*oneobj).at(2);
            float h = (*oneobj).at(3);
            // x[:, 5:] *= x[:, 4:5]
            // 参照yolov5源代码： x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf 
            // 相对大小不变我认为可以不用相乘
            float conf =  (*oneobj).at(4);
            // float *xyxy = xywh2xyxy(x, y, w, h);
            float left = 0;
            float right = 0;
            xywh2lrwh(x, y, w, h, left, right);
            auto index = max_element((*oneobj).begin()+5, (*oneobj).end());
            float maxvalue = *index;
            int classId = index - (*oneobj).begin() - 5;
            // printf("类别编号为： %d\n", classId);
            // printf("conf为： %f\n", conf);
            // float cls = （float)(index - (*oneobj).begin() - 5);
            // vector<float> f = {x, y, w, h, conf, cls};
            // filterout.push_back(f);

            boxes.emplace_back(left*r_w, right*r_h, w*r_w, h*r_h);
            confs.emplace_back(conf);
            classIds.emplace_back(classId);
            // cout << left << "  " << right <<"  " <<  w << "  " << h << "  " << conf << "  " << classId << endl;
        }
    }
    // nms
    // py_cpu_nms(boxes, scores, iou_thres);
    // cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, conf_thres, iou_thres, indices);
    printf("绘制结果中...... \n");
    printf("检测到%d个目标...... \n", indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confs[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
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
    // int device_type = kDLCUDA;
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
    
    // 计时
    clock_t start, finish;
    double   duration;
    start = clock();
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

    start = clock();
    run();
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "推理耗时：" << duration << "s" << endl;
    get_output(0, y);

    auto result = static_cast<float*>(y->data);
    vector<vector<float>> output;

    for(int i = 0; i < out_shape[1]; i++){
        vector<float> oneobj;
        for(int j = 0; j < out_shape[2]; j++){
            // cout<<result[i*85+j]<<endl;
            oneobj.push_back(result[i*85+j]);
        }
        output.push_back(oneobj);
    }

    // 后处理 nms  draw
    non_max_suppression(image, out_shape, output, 0.25, 0.3);

    // 
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "总耗时：" << duration << "s" << endl;
    // 推理耗时：1.61561s
    // 绘制结果中...... 
    // 检测到5个目标...... 
    // 总耗时：1.97885s
}
