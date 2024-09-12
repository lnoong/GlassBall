#### 一、准备工作

1. 下载yolov5[源码](https://github.com/ultralytics/yolov5)

![2](C:\Users\work\Desktop\yolocpp\2.png)

2. 解压工程

   解压后进入data文件夹，在data目录下新建datasets文件夹，这个文件夹存放要用到的数据集。scripts目录里有自动化脚本，可以下载网上的现成的数据集有coco及coco128等，双击运行后也会自动创建datasets文件夹，这里我们选择自己创建。

![5](C:\Users\work\Desktop\yolocpp\5.png)

![4](C:\Users\work\Desktop\yolocpp\4.png)

3.进入datasets文件夹，创建自己数据集的文件夹mydataset，在mydataset里创建images和labels文件夹，images存放图片，labels存放打标的信息文件，将待打标的图片放入images。

![](C:\Users\work\Desktop\yolocpp\6.png)

4.整个文件目录结构如下图所示

![](C:\Users\work\Desktop\yolocpp\7.png)

#### 二、使用labelImg标注图片

1. 修改labelImg安装目录下的predefined_classes.txt文件，将自己用到的标签输入，每行一个。

   ![](C:\Users\work\Desktop\yolocpp\8.png)

2. 打标，labelImg快捷键，A上一张图片，D下一张图片，W创建新方框，打完记得保存。

   ![](C:\Users\work\Desktop\yolocpp\9.png)

#### 三、配置文件修改

1. 修改配置文件并划分训练集、验证集、测试集

   在data目录下新建文本文件，改为yaml后缀，使用记事本打开，输入以下内容：

   ```yaml
   # mydataset
   
   # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
   path: .../data/datasets/mydataset # dataset root dir
   train: images/  # train images (relative to 'path') 128 images
   val: images/  # val images (relative to 'path') 128 images
   test:  # test images (optional)
   
   # Classes
   nc: 1  # number of classes
   names: ['tom']  # class names
   
   ```

2. 下载模型并修改模型配置文件

   根据需要去官方仓库下载模型，这里选用yolov5s，下载到models文件夹

   ![](C:\Users\work\Desktop\yolocpp\11.png)

   修改models文件夹下的yolov5s.yaml，如图，只需修改nc的参数，这里只有一个标签，所以改为1

![](C:\Users\work\Desktop\yolocpp\12.png)

#### 四、模型训练

1. 打开yolov5-master 目录下的 train.py 程序，修改如下三个路径，运行即可

   ![](C:\Users\work\Desktop\yolocpp\13.png)

2.根据不同情况，可能需要更改其他参数，下面是我可用的参数（因人而异）

```python
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'models/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/mydataset.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()
```



#### 五、报错处理

1. Bad git executable.

   - 原因：下载源码使用的是网页，没有使用git clon

   ![](C:\Users\work\Desktop\yolocpp\err\1.png)

   - 解决方法：

     ![](C:\Users\work\Desktop\yolocpp\err\2.png)

     ```python
     os.environ["GIT_PYTHON_REFRESH"] = "quiet"
     ```

2.  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

   - 原因：正在初始化 libiomp5md.dll，但发现 libiomp5md.dll 已经初始化。（多个libiomp5md.dll）
   - 解决方法：libiomp5md.dll [解决方法总结](https://blog.csdn.net/zhuma237/article/details/128271897)

