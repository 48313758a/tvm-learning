---

---

# tvm-learning



![res](https://raw.githubusercontent.com/48313758a/tvm-learning/main/res.jpg)

This PythonAPI infer result



![C_Res](https://raw.githubusercontent.com/48313758a/tvm-learning/main/C_Res.jpg)

This is C++API infer result. 



RTX2070s CUDA infer cost 0.001s per image(640x640).

CPU costs 1.6s.



C++部署:onnx->params+json :

​       yolo5n cpu+float32 0.3s 并且cuda+float32 0.4ms ， 

​       yolo5s ： cpu+float32 1.5s并且cuda+float32 1ms

由于是float32cpu推理较慢，可用tflite int8格式加速，暂时无数据集进行量化训练。
