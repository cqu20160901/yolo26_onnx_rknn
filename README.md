# yolo26 部署

&emsp;&emsp; 2026年1月15日yolo26 开源了，基于对yolo部署尚且还有的些许热情，还是继续部署搞一下。从2020年yolov5开始，到2025年的yolov13，再到今年2026年的yolo26，用或没有用过每个版本都部署过。自己实际使用的v5、v6、v7、v8，加上这两年没有怎么做单目2D目标检测，从v9及其之后的版本就属于兴趣驱使的去学习了解。

完整代码：包括onnx转rknn和测试代码、rknn板端部署C++代码

[【onnx转rknn和测试代码】](https://github.com/cqu20160901/yolo26_onnx_rknn)

[【rknn板端部署C++代码】](https://github.com/cqu20160901/yolo26_rknn_Cplusplus)

# 1 模型训练

&emsp;&emsp;yolo26 训练参考官方代码。

# 2 导出 yolo26 onnx

&emsp;&emsp; 导出onnx修改以下几处，和之前的导出yolov8、yolov11、yolov12、yolov13类似，修改输出头代码，增加保存onnx的代码。之前有网友留言问为啥不能基于官方提供的导出onnx，用官方代码导出的onnx对部署板端芯片说时候通用性（可能部分板端芯片不支持后处理中的部分操作）和性能（对部分板端芯片后处理操虽然也在转换成功了，但可能是用cpu做的）不是很友好。

第一处：修改导出onnx的检测头

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dcd45d75d6184408a55762c94514e7ec.png)

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


```python
        # 导出 onnx 增加
        return one2one
```

&emsp;&emsp; 在败部署时这里第一次导出错了，用的one2many，推理onnx结果有很多框（像是没有进行nms的效果），于是把nms加上检测结果看着很正常（心中满是疑惑和奇怪，端到端的怎么需要nms）；还有就是onnx的可视化结果和pytorch结果不一致，猜测权重加载错了，于是测试了官方导出的onnx结果和pytorch结果一致，可以确认这一步哪里有问题；于是return one2one导出onnx，测试结果正常，不需要nms，且和pytorch一致。

第二处：增加保存onnx代码

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/713bc1efd9df43cd8206e4377315a623.png)

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

&emsp;&emsp; 有网友曾问按照博客导出的onnx，用博客中的代码测试结果不正常，而用博客中提供的onnx结果正常，是由于推理onnx时模型输出的顺序不是"output1、output2、output3、output4、output5、output6"这个顺序，可能是"output2、output1、output4、output3、output6、output5"，后处理代码取的时候调整一下即可，特别是上芯片后的顺序变的可能性更大。最好的办法是打印一下维度，根据输出维度调整一下取值的顺序。


修改完以上两处，运行以下代码：

```python
from ultralytics import YOLO
model = YOLO(model='yolo26n.pt')  # load a pretrained model (recommended for training)
results = model(task='detect', source='./test.jpg', save=True)  # predict on an image
```

***特别说明：*** 修改完以上两处运行可能会会报错，但不影响onnx的生成；生成onnx后强烈建议用from onnxsim import simplify 处理一下再转rknn。

# 3 测试onnx效果

pytorch效果

![在这里插入图片描述](https://github.com/cqu20160901/yolo26_onnx_rknn/blob/main/yolo26n_onnx/test_pytorch_result.jpg)

onnx效果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e94e48ed4562445b93facfdc5ec03214.png)

pytorch效果和onnx效果是一致的。

# 4 onnx转rknn

onnx转rknn[代码链接](https://github.com/cqu20160901/yolo26_onnx_rknn)

转rknn后仿真结果

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e25c03b719b14e2d835c3f5420734d64.jpeg)

# 5 rk3588板子测试yolo26模型

使用的 rknn_toolkit 版本：rknn_toolkit2-2.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
C++代码中的库和工具链的版本注意对应适配。


1）编译

```
cd examples/rknn_yolo26_open

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
    char model_path[256] = "/home/zhangqian/rknn/examples/rknn_yolo26_demo_open/model/RK3588/yolo26n_80class_ZQ.rknn";
    char image_path[256] = "/home/zhangqian/rknn/examples/rknn_yolo26_demo_open/test.jpg";
    char save_image_path[256] = "/home/zhangqian/rknn/examples/rknn_yolo26_demo_open/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```

3）板端效果和时耗

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ffbfce0e3e864266a9aecb06f534ad18.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c20982eb8974782afcec33ed3bba427.png)

&emsp;&emsp;后处理中代码对于输出80个类别，同样先选择最大类别值，然后再进行反量化，最后只对这一个类别值进行sigmoid，这样后处理的时耗加速很多。最初也是按照对类别每个输出值都反量化和sigmoid，再判断是否大于阈值，有一定的优化。

