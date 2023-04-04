# yolov5_d435i_detection
**使用realsense d435i相机，基于pytorch实现yolov5目标检测，实时返回检测目标相机坐标系下的位置信息。**

# 1.Environment：

1.一个可以运行YOLOv5的python环境

```bash
pip install -r requirements.txt
```

2.一个realsense相机和pyrealsense2库

```bash
pip install pyrealsense2
```

**在下面两个环境中测试成功**

- **win10** python 3.8 Pytorch 1.10.2+gpu CUDA 11.3  NVIDIA GeForce MX150

- **ubuntu16.04**  python 3.6 Pytorch 1.7.1+cpu

# 2.Results：

- Colorimage:

![image-20220213144406079](https://github.com/Thinkin99/yolov5_d435i_detection/blob/main/image/image-20220213144406079.png)

- Colorimage and depthimage:

![image-20220213143921695](https://github.com/Thinkin99/yolov5_d435i_detection/blob/main/image/image-20220213143921695.png)

# 3.Model config：

修改模型配置文件，这里以yolov5s模型为例。也可以使用自己训练的权重模型。

```yaml
weight:  "weights/yolov5s.pt"
# 输入图像的尺寸
input_size: 640
# 类别个数
class_num:  80
# 标签名称
class_name: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
# 阈值设置
threshold:
  iou: 0.45
  confidence: 0.6
# 计算设备
# - cpu
# - 0 <- 使用GPU
device: '0'
```

# 4.Camera config：

分辨率好像只能改特定的参数，不然会报错。d435i可以用 1280x720, 640x480, 848x480。

```python
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
```
# 5.code return xyz：
下方代码实现从像素坐标系到相机坐标系转换，并且标注中心点以及三维坐标信息。
```python
for i in range(len(xyxy_list)):
    ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2)  # 计算像素坐标系的x
    uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2)  # 计算像素坐标系的y
    dis = aligned_depth_frame.get_distance(ux, uy)  
    camera_xyz = rs.rs2_deproject_pixel_to_point(
    depth_intrin, (ux, uy), dis)  # 计算相机坐标系xyz
    camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
    camera_xyz = camera_xyz.tolist()
    cv2.circle(canvas, (ux,uy), 4, (255, 255, 255), 5)#标出中心点
    cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#标出坐标
    camera_xyz_list.append(camera_xyz)
    #print(camera_xyz_list)
```
# 6.Reference:

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

[https://github.com/mushroom-x/yolov5-simple](https://github.com/mushroom-x/yolov5-simple)
