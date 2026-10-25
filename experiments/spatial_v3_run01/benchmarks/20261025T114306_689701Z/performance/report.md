# 雾密度模块工程性能

执行状态：PASS。计时精度 FP32，batch=1；预热和重复次数见 JSON。

端到端包含内存 PNG 解码、RGB 转换/缩放、张量转换、传输、前向和回传，不含磁盘、网络、模型初始化。吞吐是该串行 batch=1 协议下的实测吞吐。

| 方法 | 设备 | 尺寸 | 范围 | 状态 | P50 ms | P95 ms | images/s | 参数量 |
|---|---|---:|---|---|---:|---:|---:|---:|
| dcp | cpu | 256 | forward | PASS | 67.881 | 79.660 | 14.269 | 0 |
| dcp | cpu | 256 | end_to_end | PASS | 70.838 | 80.827 | 13.907 | 0 |
| prior | cpu | 256 | forward | PASS | 503.286 | 510.470 | 1.987 | 0 |
| prior | cpu | 256 | end_to_end | PASS | 512.022 | 516.678 | 1.956 | 0 |
| student | cpu | 256 | forward | PASS | 384.725 | 403.119 | 2.584 | 864752 |
| student | cpu | 256 | end_to_end | PASS | 393.581 | 417.494 | 2.537 | 864752 |
| dcp | cpu | 512 | forward | PASS | 273.807 | 289.661 | 3.616 | 0 |
| dcp | cpu | 512 | end_to_end | PASS | 281.473 | 285.918 | 3.552 | 0 |
| prior | cpu | 512 | forward | PASS | 2061.885 | 2099.983 | 0.484 | 0 |
| prior | cpu | 512 | end_to_end | PASS | 2086.426 | 2118.417 | 0.479 | 0 |
| student | cpu | 512 | forward | PASS | 1776.704 | 1847.136 | 0.562 | 864752 |
| student | cpu | 512 | end_to_end | PASS | 1772.254 | 1821.802 | 0.564 | 864752 |
| dcp | cuda | 256 | forward | PASS | 0.168 | 0.175 | 5916.565 | 0 |
| dcp | cuda | 256 | end_to_end | PASS | 3.538 | 5.676 | 246.583 | 0 |
| prior | cuda | 256 | forward | PASS | 2.964 | 2.996 | 337.015 | 0 |
| prior | cuda | 256 | end_to_end | PASS | 6.384 | 6.415 | 156.621 | 0 |
| student | cuda | 256 | forward | PASS | 9.563 | 9.690 | 104.540 | 864752 |
| student | cuda | 256 | end_to_end | PASS | 12.918 | 13.120 | 77.223 | 864752 |
| dcp | cuda | 512 | forward | PASS | 0.334 | 0.338 | 2986.632 | 0 |
| dcp | cuda | 512 | end_to_end | PASS | 12.379 | 13.052 | 80.025 | 0 |
| prior | cuda | 512 | forward | PASS | 2.672 | 2.681 | 374.205 | 0 |
| prior | cuda | 512 | end_to_end | PASS | 10.694 | 10.789 | 93.313 | 0 |
| student | cuda | 512 | forward | PASS | 26.355 | 26.578 | 37.931 | 864752 |
| student | cuda | 512 | end_to_end | PASS | 35.067 | 36.605 | 28.392 | 864752 |

显存与原始逐次延迟见 `performance.json`。CPU 显存为 N/A；CUDA 缺失为 NOT RUN。MACs 仅统计 Conv/Linear，不能冒充完整 FLOPs。

`student_architecture_only` 使用未训练的相同网络，仅反映结构计算成本，其正确性仍是 NOT EVALUATED。设备不同的数字不能直接证明算法优劣。
