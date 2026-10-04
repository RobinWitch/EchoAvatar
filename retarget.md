下面重定向操作在windows系统上执行，要求下载安装好MotionBuilder。

以motion builder的方式来批量重定向动作数据作为案例。请确保你已经安装了MotionBuilder。并且已经下载解压好
[zeroeggs-retarget-fbx](https://theorangeduck.com/media/uploads/Geno/zeroeggs-retarget/fbx.zip)
[motorica-retarget-fbx](https://theorangeduck.com/media/uploads/Geno/motorica-retarget/fbx.zip)
在此项目中，我们使用这个[AX_female21](tools/AX_female.7z)作为目标样例角色,使用前请先解压

& "D:\MotionBuilder 2026\bin\x64\mobupy.exe" `
    "/path/to/tools/motionbuilder_retarget_batch.py" `
    --target-fbx "/path/to/tools/AX_female/AX_female2.fbx" `
    --source-dir "/path/to/motorica/fbx" `
    --source-type motorica `
    --output-dir "/path/to/motorica/retarget_AX_female21" `
    --overwrite


& "D:\MotionBuilder 2026\bin\x64\mobupy.exe" `
    "/path/to/tools/motionbuilder_retarget_batch.py" `
    --target-fbx "/path/to/tools/AX_female/AX_female2.fbx" `
    --source-dir "/path/to/zeroeggs/fbx" `
    --source-type zeroeggs `
    --output-dir "/path/to/zeroeggs/retarget_AX_female21" `
    --overwrite

提取其中的bvh文件夹，audio文件请自行在zeroeggs以及motorica官方仓库中下载，下载后将文件均移至服务器的datasets/zm/wav路径下

将两个重定向的output文件夹中所有文件均移动到服务器的datasets/zm/raw路径下

接下来使用 `tools/prepare_bvh_format.py` 一次完成关节过滤、坐标转换、初始位置与朝向统一，以及镜像增强：

```bash
python tools/prepare_bvh_format.py \
    --input-dir datasets/zm/raw \
    --output-dir datasets/zm/all \
    --workers 8
```

脚本会按以下顺序处理：

1. 将所有 BVH 从 355 个关节过滤为 88 个关节。
2. 绕 X 轴旋转 +90°，将 Y-up 转换为 Z-up，并同步转换骨骼偏移、根节点位置和关节旋转。输出欧拉旋转通道顺序直接设定为：Motorica（文件名以 `kth` 开头）使用 `ZYX`，其余 ZeroEGGS 文件使用 `XYZ`。
3. 对所有 ZeroEGGS 和 Motorica 数据，将首帧根节点的水平位置 X、Y 归零，保留 Z 高度，并消除初始绕 Z 轴的朝向角；整段动作与轨迹同步调整。
4. 仅对文件名以 `kth` 开头的 Motorica 数据，在归一化后交换左右关节并沿 X 方向反射，生成对应的 `原文件名_mirror.bvh`。

原始数据保留在 `datasets/zm/raw`，处理结果写入 `datasets/zm/all`。默认跳过已有输出，需要重新生成时添加 `--overwrite`。

按划分列表，将 `datasets/zm/all` 下处理后的 BVH 文件复制到 `datasets/zm/train` 和 `datasets/zm/valid` 中：

```bash
rsync -a --ignore-missing-args --files-from=./datasets/zm/train.txt ./datasets/zm/all ./datasets/zm/train
rsync -a --ignore-missing-args --files-from=./datasets/zm/valid.txt ./datasets/zm/all ./datasets/zm/valid
```
