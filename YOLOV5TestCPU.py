import onnx
import numpy as np
from PIL import Image
from tvm import relay, autotvm
from tvm.relay import testing
import tvm
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
import coremltools
import tensorflow as tf
import tflite
# from tvm import te
# from tvm.contrib import graph_executor
# import tvm.testing




img = Image.open('./dog.jpg').resize((640, 640))
dtype = 'float32'



# onnx model
onnx_model = onnx.load('./yolov5n.onnx')
img = np.array(img).transpose((2, 0, 1)).astype(dtype)
input_name = 'images'
# coreml model
# coreml_model = coremltools.models.MLModel('./yolov5s.mlmodel')
# tflite model
# buf = open('./yolov5s-fp16.tflite', 'rb').read()
# tflite_model = tflite.Model.GetRootAsModel(buf, 0)
# tflite shape
# img = np.array(img).astype(dtype)
# input_name = 'input_1'  # change '1' to '0'


img = img/255.0    # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

log_file = "history.log"
# 这里首先在PC的CPU上进行测试 所以使用LLVM进行导出
# target = tvm.target.create('llvm')
target = tvm.target.Target('llvm')
# input_name = 'images'  # change '1' to '0'
# input_1 Identity

shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
# sym, params = relay.frontend.from_coreml(coreml_model, shape_dict)
# sym, params = relay.frontend.from_tflite(tflite_model, shape_dict)


##################################################
# number = 10
# repeat = 1
# min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
# timeout = 10  # in seconds
# # create a TVM runner
# runner = autotvm.LocalRunner(
#     number=number,
#     repeat=repeat,
#     timeout=timeout,
#     min_repeat_ms=min_repeat_ms,
#     enable_cpu_cache_flush=True,
# )
# tuning_option = {
#     "tuner": "xgb",
#     "trials": 10,
#     "early_stopping": 100,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(build_func="default"), runner=runner
#     ),
#     "tuning_records": log_file,
# }
# # begin by extracting the taks from the onnx model
# tasks = autotvm.task.extract_from_program(sym["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),))
# # Tune the extracted tasks sequentially.
# for i, task in enumerate(tasks):
#     prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
#     tuner_obj = XGBTuner(task, loss_type="rank")
#     tuner_obj.tune(
#         n_trial=min(tuning_option["trials"], len(task.config_space)),
#         early_stopping=tuning_option["early_stopping"],
#         measure_option=tuning_option["measure_option"],
#         callbacks=[
#             autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
#             autotvm.callback.log_to_file(tuning_option["tuning_records"]),
#         ],
#     )
##################################################
# from tvm.nnvm.to_relay import to_relay
# sym, params = to_relay(sym, shape_dict, 'float32', params=params)

# print("optimize relay graph...")
# with tvm.relay.build_config(opt_level=2):
#     sym = tvm.relay.optimize(sym, target, params)

# quantize
# print("apply quantization...")
# from tvm.relay import quantize
# with quantize.qconfig(nbit_input=8,
#                     nbit_weight=8,
#                     global_scale=8.0,
#                     dtype_input='int8',
#                     dtype_weight='int8',
#                     dtype_activation='int8',):
#     # sym['main'] = tvm.relay.quantize(sym['main'],params=params)
#     sym = quantize.quantize(sym, params)
    

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(sym, target, params=params)

##################################################
from tvm.contrib import graph_runtime, graph_executor

# 下面的函数导出我们需要的动态链接库 地址可以自己定义
print("Output model files")
libpath = "./tvm_output_lib/yolov5s.so"
lib.export_library(libpath)

# 下面的函数导出我们神经网络的结构，使用json文件保存
graph_json_path = "./tvm_output_lib/yolov5s.json"
with open(graph_json_path, 'w') as fo:
    fo.write(graph)

# 下面的函数中我们导出神经网络模型的权重参数
param_path = "./tvm_output_lib/yolov5s.params"
with open(param_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))
# -------------至此导出模型阶段已经结束--------

##################################################
# 接下来我们加载导出的模型去测试导出的模型是否可以正常工作
loaded_json = open(graph_json_path).read()
# loaded_lib = tvm.module.load(libpath)
loaded_lib = tvm.runtime.load_module(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

# 这里执行的平台为CPU
ctx = tvm.cpu()

# module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module = graph_executor.create(loaded_json, loaded_lib, ctx)

module.load_params(loaded_params)
module.set_input(input_name, x)
module.run()
pred = module.get_output(0).asnumpy()

print(pred.shape)