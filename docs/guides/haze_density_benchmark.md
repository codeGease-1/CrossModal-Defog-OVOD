# 独立雾密度 Benchmark 使用说明

阶段 2 只评价雾密度模块，不训练、不去雾、不做目标检测。先阅读 [方案](../benchmark_plan.md) 和 [实际科研审查报告](../stage_2_research_review.md)。

## 给初学者的解释

现有系统先用暗通道、局部对比度和颜色差异三条规则制作一张“看起来像雾”的评分图，再训练学生网络模仿它。学生模仿得像，只能证明学会了老师的规则，不能证明老师判断正确。

本次另找清晰遥感图，按 `I = J*t + A*(1-t)` 人为加雾：`J` 是清晰图，`A` 是大气光，`t` 越小，保留的场景信号越少，加入的雾幕越多。我们知道自己加了多少，因此用独立的 `1-t` 检查预测排序。没有用三先验自己出的答案来给三先验打分。

需要分开问三个问题：程序是否按设计计算；雾浓淡判断是否符合独立参照；计算要花多少时间。测试通过只回答第一个问题。本次发现三先验擅长的图内空间分布，与其失败的跨图雾强度排序，是两种不同能力。

## VS Code + Colab

你现有的 [最简实验流程 notebook](../雾密度网络_最简实验流程.ipynb) 已在**第 8 节**集成此 Benchmark。第 1、3、4 节准备好项目、RSHaze+ 与实验目录后即可运行；已有当前版本 `best.pth` 时自动加入学生网络。输出存到本次实验的 Google Drive `benchmarks/<时间戳>/`，第 8 节最后一个单元直接显示主要表格与示例图。评测代码作为快照内嵌，不依赖远端分支已经上传未提交的 Benchmark 文件。变更评测源码后需先运行 `python scripts/benchmark/build_colab_notebook.py` 和 `python scripts/benchmark/sync_minimal_notebook.py` 更新两个 notebook。

也可只打开 [独立运行 notebook](../haze_density_benchmark_colab.ipynb)，右上角选 Colab 内核，依次运行三个单元。它无需完成训练流程；评测代码同样内嵌，不依赖 Colab 能看到 Windows 的 D 盘。

入口先查常见数据路径；缺少匹配样本时显式使用 `--fetch-small-sample` 从官方 ZIP 按字节范围读取少量图片，硬限制 24 MiB，拒绝服务器返回整个压缩包。本机这次实际读取 8,622,736 字节。不会下载完整 7.38 GB 数据集。如果服务器不支持范围读取会报错，需要指定自己的 `--data-root`。

默认不猜测权重。已有修复后训练权重时，在 notebook 第二个单元的命令中加 `--checkpoint /实际路径/best.pth`。检查内部 `prior_version=spatial_minmax_valid_boundary_v3` 与完成的训练 epoch；旧权重不能当作 V1 正确性结果。还需人工核对训练场景与测试场景隔离，元数据本身不能证明没有数据泄漏。

结果保存在打印的 `/content/stage2_haze_.../experiments/colab_benchmark_...` 下；关闭 Colab 前打包保存。notebook 没有预填结果，本次本地运行也不代表已在你的 Colab GPU 执行。

## 本地分步命令

在仓库根目录使用已安装 torch、numpy、Pillow、PyYAML、matplotlib 的 Python。可独立运行任一步，已有非空生成/评价目录会被拒绝覆盖。

```bash
python -m unittest scripts.test.test_haze_benchmark scripts.test.test_stage1_bugfixes scripts.test.test_review_regressions -v
# 复现固定四 ROI 试验，优先复用本地官方图像；缺失时允许有限小样本下载：
python scripts/benchmark/run_colab_benchmark.py --fetch-small-sample
# 自己准备 manifest 后逐步执行：
python -m benchmarks.haze_density generate --scenes path/to/scenes.json --output experiments/run1/data --profile minimal
python -m benchmarks.haze_density evaluate --dataset experiments/run1/data --output experiments/run1/results --device cpu
python -m benchmarks.haze_density report --output experiments/run1/results --max-figures 32
python -m benchmarks.haze_density performance --output experiments/run1/performance --devices cpu cuda
```

将 `--profile minimal` 换为 `full` 得到全部 A、亮度和空间模式；更换输出目录保存两套结果。默认配置为 [haze_benchmark.yaml](../../configs/haze_benchmark.yaml)，自定义使用 `python -m benchmarks.haze_density --config your.yaml ...`。生成与评价应使用同一份配置。

固定试验清单在 [pilot manifest](../../configs/haze_benchmark_pilot_scenes.json)。它的相对图像路径需相对于清单所在地；入口会复制清单到实际数据目录。自己的数据可执行 `discover --root 数据目录 --output scenes.json --limit 8`，只搜索官方 test/cleanpng；自动发现的地物标记为 unknown、地理组保守合并，必须根据来源人工完善。

## 真值、分组与指标

每条清单记录 `scene_id/group_id/path/split/land_cover/source`，可加 ROI 与源文件 SHA256。split 为 train/calibration/test；同一地理场景的不同裁片和所有雾变体必须共用 group_id，不能跨划分。检查同组跨划分和重复图像哈希；哈希不能发现所有部分重叠裁片，仍需地理元数据复核。本次四 ROI 全属同一保守地理组，未训练、未拟合映射。

默认将普通 RGB 按 sRGB 近似解码到线性域，亮度乘数应用于 J，再按公式混合并编码回 RGB。A 的配置值属于该合成线性域。原始传感器响应、定标、多次散射、邻近效应和波段差异均未建模；不是严格的真实辐射传输仿真。非正方形 ROI 缩放为正方形也可能改变纹理尺度。`display_rgb` 可作为额外敏感性试验，必须单独标记，不与默认域混算。

每个 NPZ 分别保存 `original_rgb/clear_rgb/hazy_rgb/transmission/airlight`，推理使用浮点数组，不使用可视化 PNG。dataset.json 保存种子、模式参数、原图来源、文件哈希、配置和环境。均匀 t 为 1/.9/.75/.6/.4；空间模式含梯度、团块、四区域；全配置含两档 A、两档亮度。

| 评价 | 定义与注意事项 |
|---|---|
| Pairwise Ranking Accuracy | 同一场景、A、亮度内，所有不同雾强度两两比较；预测平局不记正确 |
| Spearman | 平局取平均秩；常量预测/真值或样本不足时返回 null + N/A，不冒充零误差 |
| 单调性违反率 | 相邻强度级出现严格反向的比例；另报平局率，防止常量输出钻空子 |
| 空间指标 | 4×4 区域平均预测与已知 1-t 的相关、排序、重雾/轻雾区域响应差 |
| 无雾响应 | t=1 的均值、分位数及参考阈值面积；暗区、亮区、低纹理区附加诊断 |
| 可视化残差 | 区域归一化秩之差，无量纲，不是物理透射率误差 |

原始 DCP 与三先验数值含义不同，因此比较排序/相关，不按响应幅值判优劣。阈值 0.7 只是各自内部诊断，不能宣称是标定后的误报率。未拟合物理映射，所以物理 MAE 为 N/A；以后需要映射时必须用独立校准组确定后冻结，再用于测试集。

报告按地物、A×亮度条件、空间模式分层；先组内平均再跨组宏平均。常量 t 不参加空间相关。固定 0～1 预测热力图、固定 t 色轴，禁止逐张自动归一化制造跨图可比性。

## 性能与状态

默认 FP32、batch=1、256/512，预热 3 次、计时 20 次、torch 单线程。GPU 每次计时前后同步；分别报告驻留设备张量前向和“内存 PNG 解码→RGB/缩放→浮点张量→传输→前向→回传 CPU”。端到端不含磁盘、网络和模型初始化。吞吐为这一串行协议的 images/s，不是最大批处理吞吐。

CPU 显存为 N/A；GPU 不存在为 NOT RUN。参数量为真实结构统计，MACs 只覆盖 Conv/Linear，明确不是完整 FLOPs。缺少学生权重时可用同结构随机权重测计算成本，方法名为 `student_architecture_only`，其预测正确性仍为 NOT EVALUATED。不得把 CPU 和 GPU 时间直接混在一起得出算法快慢结论。

PASS = 该项成功执行且满足形状/有限值等实现检查；FAIL = 执行或校验错误；NOT RUN = 缺少所需资源；N/A = 指标数学上或设备上不适用。没有预注册正确性验收门槛，因此科学指标是描述性数值，不把执行 PASS 叫作模型合格。

## 输出与可复现记录

`results.json` 是完整评价记录，CSV 包含总体/地物、条件分层、空间模式、逐图和排序表；`report.md` 与 `figures/` 是自动报告。性能目录含逐次延迟、P50/P95、显存、参数量和局部 MACs。Git commit、工作区状态、源文件 SHA256、配置、Python/PyTorch/CPU/GPU 信息均记录；notebook 另保存代码包来源 `bundle_origin.json`。改变代码后运行 `python scripts/benchmark/build_colab_notebook.py` 重建 notebook。

真实图像辅助评价见 [盲评协议](../benchmark_human_protocol.md)。尚无独立真实物理标注，因此本次不声称已验证真实遥感雾浓度。
