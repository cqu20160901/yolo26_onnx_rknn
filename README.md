# yolo26 部署。

完整代码：包括onnx转rknn和测试代码、rknn板端部署C++代码

[【onnx转rknn和测试代码】](https://github.com/cqu20160901/yolo26_onnx_rknn)

[【rknn板端部署C++代码】](https://github.com/cqu20160901/yolo26_rknn_Cplusplus)

# 1 模型训练

&emsp;&emsp;yolo26 训练参考官方代码。

# 2 导出 yolo26 onnx

导出onnx修改以下几处。

第一处：修改导出onnx的检测头

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45b36003770a4b2bb2d70e194b830193.png)

```python
        # 导出 onnx 增加
        y = []
        for i in range(self.nl):
            t1 = box_head[i](x[i])
            t2 = cls_head[i](x[i])
            y.append(t1)
            y.append(t2)
        return y
```

第二处：增加保存onnx代码

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/064306acddd2429fb1cfafeb93afe47f.png)


```python
        print("===========  onnx =========== ")
        import torch
        dummy_input = torch.randn(1, 3, 640, 640)
        input_names = ["data"]
        output_names = ["output1", "output2", "output3", "output4", "output5", "output6"]
        torch.onnx.export(self.model, dummy_input, "/root/autodl-tmp/zhangqian/yolo26/ultralytics/yolov26n_onnx/yolo26_80class_ZQ.onnx",
                          verbose=False, input_names=input_names, output_names=output_names, opset_version=11)
        print("======================== convert onnx Finished! .... ")
```

修改完以上两处，运行以下代码：

```python
from ultralytics import YOLO
model = YOLO(model='yolo26n.pt')  # load a pretrained model (recommended for training)
results = model(task='detect', source='./test.jpg', save=True)  # predict on an image
```

***特别说明：*** 修改完以上两处后运行会报错，但不影响onnx的生成；生成onnx后强烈建议用from onnxsim import simplify 处理一下再转rknn。

# 3 测试onnx效果

pytorch效果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b866a799d3584c8f9e6e83bde03a5997.jpeg)

onnx效果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b7e935734b0942d2b309508f09053a38.jpeg)
# 4 onnx转rknn
onnx转rknn[代码链接](https://github.com/cqu20160901/yolo26_onnx_rknn/tree/main/yolo26n_rknn)

转rknn后仿真结果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e25c03b719b14e2d835c3f5420734d64.jpeg)

# 5 rk3588板子测试yolov11模型

使用的 rknn_toolkit 版本：rknn_toolkit2-2.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
C++代码中的库和工具链的版本注意对应适配。


1）编译

```
cd examples/rknn_yolo26_demo_dfl_open

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_yolo_demo_Linux

./rknn_yolo_demo 

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/zhangqian/rknn/examples/rknn_yolo26_demo_dfl_open/model/RK3588/yolo26n_80class_ZQ.rknn";
    char image_path[256] = "/home/zhangqian/rknn/examples/rknn_yolo26_demo_dfl_open/test.jpg";
    char save_image_path[256] = "/home/zhangqian/rknn/examples/rknn_yolo26_demo_dfl_open/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```

3）板端效果和时耗

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eefc9e3da1084cf1822204f3b0d0af88.jpeg)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/897e9840ea334b6ea0ece709b900f8af.png)

这个时耗比较长，毫无疑问很多算子切换到cpu上进行的。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf1aec2a959d4333bf180f5058b3b1ae.png)
