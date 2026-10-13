# TangDaoWan

唐岛湾滨海湿地地物分类 —— 基于 GF-2 卫星影像与无人机（UAV）影像的语义分割实验代码。

## 任务

把滨海湿地地表划分为 5 类：

| 标签 | 类别 |
| --- | --- |
| 0 | `_background_` 背景 |
| 1 | Emergent plant 挺水植物 |
| 2 | Vegetation 植被 |
| 3 | Pit and pond 坑塘 |
| 4 | Muddy beach 泥滩 |

## 数据

| 文件 | 说明 |
| --- | --- |
| `16GF-2.tif` ~ `23Gf-2.tif` | 高分二号（GF-2）卫星影像，逐年 |
| `UAV_RGB.tif` / `UAV_NIR.tif` | 无人机可见光 / 近红外影像 |
| `data/**/*.npy` | 切片后的训练、验证数据与标签（未入库） |

> 影像文件用 Git LFS 管理（见 `.gitattributes`），克隆后需先 `git lfs pull`。

## 方法

- **LinkResNet**：以 ImageNet 预训练的 ResNet34 作编码器，配对称的转置卷积解码器，
  并在各级解码处与编码特征相加（残差连接），输出 5 类；
- **随机森林**：作为对比基线，`n_estimators=100`，用混淆矩阵评估各类精度与召回率；
- `*_att.py`：在对应训练脚本基础上的注意力机制变体。

## 脚本

| 脚本 | 作用 |
| --- | --- |
| `S_train.py` / `UAV_train.py` | 卫星影像 / 无人机影像训练 |
| `S_train_att.py` / `UAV_train_att.py` | 对应的注意力变体 |
| `S_Retrain.py` | 重新训练 |
| `S_RF.py` / `UAV_RF.py` | 随机森林基线，输出混淆矩阵与各类精度 |
| `pre.py` | 推理：逐 patch 预测 → 拼回整幅（5376×4864）→ 类别着色并输出 tif |

## 环境

```
python >= 3.8

torch
torchvision
tifffile
numpy
scikit-learn
seaborn
matplotlib
```

## 使用

1. 将遥感影像切片、标注后存为 `.npy`，并放到各脚本里 `dataset_path` 指向的位置；
2. 训练：`python S_train.py`（或 `python UAV_train.py`）；
3. 预测：`python pre.py`，结果输出到 `data/all/23year/23pre.tif`。

> 脚本中的数据路径与权重路径均为本机绝对/相对路径，运行前请按实际位置调整。

---

> 本 README 依据仓库现有代码整理，细节如与实现不符请以代码为准。
