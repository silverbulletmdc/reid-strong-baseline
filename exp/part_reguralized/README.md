# Part Reguralized

复现论文《Part-regularized Near-duplicate Vehicle Re-identification》

## Setup

首先初始化环境

```bash
sh init_env.sh
```

### Prepare box data

训练部分基本上使用了YOLOV3的原工程。具体操作细节可以参见[YOLO官网](https://pjreddie.com/darknet/yolo/)。

将整个数据集的图片都使用YOLOv3分别检测车窗和车灯(作者并未公布brand的标注)。

生成结果文件为`veri776_with_box.json`


### Train



