# yolo

## 进度
6.18 Sat 基础工程监考
6.19 Sun 桥梁工程监考
6.20 Mon 桥梁工程批卷
6.21 Tue 盾构隧道ansys apdl 2D建模分析
6.22 Wed 转换技术路线，盾构隧道ansys apdl 2D均质圆环（惯用法）建模分析，模块化，计算结果自动输出
6.23 Thu 服务器重装esxi，建筑结构学报校稿
6.24 Fri 盾构隧道ansys apdl 3D建模分析
6.25 Sat 服务器windows server显卡直通（需设置参数256），配置CUDA+cudnn环境
6.26 Sun 服务器ubuntu配置失败（网上教程较少），改为重装centos，配置桌面
6.27 Mon 服务器centos显卡直通（卡在A100大显存特性，切换启动方式），配置CUDA+cudnn环境
6.28 Tue 农村路面开会，明确内容（服务器重装最新代码消失）重构YOLO
6.29 Wed 重构YOLO：detect与val
6.30 Thu 重构YOLO：train，完成detection 100 epoch training test（掉了6个点，可能涉及：wamrup，cosinedecay，parametergroup，ema，focalloss等）
7.01 Fri 添加grid检测头（修改mosaic等配套，修正联合损失高宽倒置和正负标签bug），完成classification + detection 100 epoch training test
7.02 Sat 完成github配置，添加rect（配套修改mosaic等）和multi_dataset，完成classification + detection 100 epoch training test
7.03 Sun 添加image_weight（multi_dataset labels 处理）和resume（cuda leaky），完成classification + detection 100 epoch training test