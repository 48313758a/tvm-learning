import onnx
import numpy as np
from PIL import Image
from tvm import relay
from tvm.relay import testing
import tvm
# from tvm import te
# from tvm.contrib import graph_executor
# import tvm.testing
import cv2

from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
#     non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
#     save_one_box
from utils.general import xywh2xyxy, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def py_cpu_nms(boxes, scores, thresh):
    """
    nms
    :param dets: ndarray [x1,y1,x2,y2,score]
    :param thresh: int
    :return: list[index]
    """
    dets = np.concatenate([boxes, scores], axis=1)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = dets[:, 4].argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1] # 不包括第0个
    return np.asarray(keep)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [np.zeros((0, 6), dtype=float)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5), dtype=float)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # print(box)
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            # conf, j = x[:, 5:].max(1, keepdim=True)
            # conf, j = np.max(x[:, 5:], axis=1, keepdims=True)
            print(x[:, 5:].shape)
            conf= np.max(x[:, 5:], axis=1, keepdims=True)
            j  = np.expand_dims(np.argmax(x[:, 5:] ,axis=1), axis=1)
            print(conf.shape)
            print(j.shape)
            # x = np.concatenate((box, conf, j), 1)[conf.view(-1) > conf_thres]
            x = np.concatenate((box, conf, j), 1)[np.squeeze(conf) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[x[:, 5:6] == classes.any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        scores = np.expand_dims(scores, axis=1)
        i = py_cpu_nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output

###################################################################################E
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

input_name = 'images'
img_path = './dog.jpg'
img_path = './bus.jpg'
img = Image.open(img_path).resize((640, 640))
img0 = cv2.imread(img_path)
# 以下的图片读取仅仅是为了测试
img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

from tvm.contrib import graph_runtime, graph_executor

# 下面的函数导出我们需要的动态链接库 地址可以自己定义
print("Output model files")
libpath = "./tvm_output_lib/yolov5s.so"

# 下面的函数导出我们神经网络的结构，使用json文件保存
graph_json_path = "./tvm_output_lib/yolov5s.json"

# 下面的函数中我们导出神经网络模型的权重参数
param_path = "./tvm_output_lib/yolov5s.params"

# 接下来我们加载导出的模型去测试导出的模型是否可以正常工作
loaded_json = open(graph_json_path).read()
# loaded_lib = tvm.module.load(libpath)
loaded_lib = tvm.runtime.load_module(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

# 这里执行的平台为CPU
ctx = tvm.cpu()
print(ctx)

# module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module = graph_executor.create(loaded_json, loaded_lib, ctx)

module.load_params(loaded_params)
# module.set_input("0", x)
module.set_input(input_name, x)
module.run()
pred = module.get_output(0).asnumpy()

print(pred.shape)
# print(pred[0][0])

pred = non_max_suppression(pred, 0.45, 0.3, None, False, max_det=1000)
for i, det in enumerate(pred):  # per image
    # annotator = Annotator(img, line_width=3, pil=not ascii)
    if len(det):
        # Rescale boxes from img_size (640, 640) to im0 size
        det[:, :4] = scale_coords((640,640), det[:, :4], img0.shape[:-1]).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            # label = None if False else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            cv2.rectangle(img0, ((int) (xyxy[0]),(int) (xyxy[1])), ((int) (xyxy[2]), (int) (xyxy[3])), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img0, names[c], ((int) (xyxy[0]), (int) (xyxy[1]) - 2), 0,  1, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    # Stream results
    # im0 = annotator.result()
    cv2.imwrite("./res.jpg", img0)
    # cv2.imshow("result", img0)
    # cv2.waitKey(-1)  # 1 millisecond
