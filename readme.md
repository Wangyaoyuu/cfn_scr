## 文件布局

训练：
train_adapt_neg_bin.py
train_normal.py  

预测：
net_predict_bin.py

参数、数据预处理和模型：
constant.py：常参数
data.py：数据的封装结构
model.py：模型
preprocess.py：预处理



Note： 

1.无论是预测还是训练，都使用了torch分布式的库。

```python
from torch.nn.parallel import DistributedDataParallel as DDP
```

2.代码中的具体路径要进一步修改。

3.data.py中的数据封装结构较复杂，建议先了解整体的训练框架。

## 文件位置

/hdd_data/wangyy_data/shrec

data：处理后的数据

data19：Shrec19数据

data20：Shrec20数据

data21：Shrec21数据

output：输出文件

paper：历年Shrec论文

src：最初版本代码



Note：数据权限可以经过sudo改为全用户共享

## 使用方法

1.数据预处理

```python
python preprocess.py 
```

在指定文件目录下生成处理后的数据，将tomo0~8的数据用hdf5格式封装至一起方便后处理。

tomo9作为测试集可单独处理，也可以合并处理。

2.训练

```python
python train_normal.py
```

不使用adaptive weight bin的方式进行训练

```
python train_adapt_neg_bin.py
```

使用adaptive weight bin的方式进行训练，使用预测的结果对数据样本采样的不同bin位置的权重进行调整。边预测边调整样本采样权重。

细节：

train_normal.py和train_adapt_neg_bin.py

在使用的数据封装结构上是有差别的，这一点需要着重注意。也是adaptive weight bin调整权重策略的核心创新点。数据的封装结构在data.py中。

3.预测

```python
python net_predict_bin.py
```

预测过程中采取gradient descent tracing的策略，不对全部的voxel进行预测，只对颗粒部分进行预测。但是这一步需要控制预测的时间，预测太多的voxel对后续的聚类不友好。

预测时间越长，预测的voxel越多，预测的颗粒也会越多。

4.聚类

```python
python cluster.py
```

通过cluster.py第一步输入的hit数值来观察预测的voxel的数目。

Note：

测试数据集的金颗粒位置是用reconstruction.mrc的density密度大小人为挑选出来的。

(金颗粒：/hdd_data/wangyy_data/shrec/output/output.submit/predict.avoid)

## Cite
```
@inproceedings{wang2024central,
  title={Central Feature Network Enables Accurate Detection of Both Small and Large Particles in Cryo-Electron Tomography},
  author={Wang, Yaoyu and Wan, Xiaohua and Chen, Cheng and Zhang, Fa and Cui, Xuefeng},
  booktitle={International Symposium on Bioinformatics Research and Applications},
  pages={212--223},
  year={2024},
  organization={Springer}
}

@Article{JCST-2409-14816,
title = {Central Feature Network Enables Accurate Detection of Both Small and Large Particles in Cryo-Electron Tomography},
journal = {Journal of Computer Science and Technology},
volume = {},
number = {},
pages = {},
year = {2025},
issn = {1000-9000(Print) /1860-4749(Online)},
doi = {10.1007/s11390-025-4816-2},	
url = {https://jcst.ict.ac.cn/en/article/doi/10.1007/s11390-025-4816-2},
author = {Yao-yu Wang and Xiao-hua Wan and Cheng Chen and Fa Zhang and Xue-feng Cui}
}
```
