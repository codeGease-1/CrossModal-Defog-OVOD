# 雾密度模块工程性能

执行状态：PASS。计时精度 FP32，batch=1；预热和重复次数见 JSON。

端到端包含内存 PNG 解码、RGB 转换/缩放、张量转换、传输、前向和回传，不含磁盘、网络、模型初始化。吞吐是该串行 batch=1 协议下的实测吞吐。

| 方法 | 设备 | 尺寸 | 范围 | 状态 | P50 ms | P95 ms | images/s | 参数量 |
|---|---|---:|---|---|---:|---:|---:|---:|
| dcp | cpu | 256 | forward | PASS | 67.725 | 82.106 | 14.247 | 0 |
| dcp | cpu | 256 | end_to_end | PASS | 70.011 | 71.167 | 14.286 | 0 |
| prior | cpu | 256 | forward | PASS | 497.313 | 502.974 | 2.007 | 0 |
| prior | cpu | 256 | end_to_end | PASS | 497.849 | 511.264 | 2.000 | 0 |
| student | cpu | 256 | forward | PASS | 377.514 | 388.372 | 2.651 | 864752 |
| student | cpu | 256 | end_to_end | PASS | 379.693 | 385.871 | 2.640 | 864752 |
| dcp | cpu | 512 | forward | PASS | 268.547 | 269.625 | 3.726 | 0 |
| dcp | cpu | 512 | end_to_end | PASS | 282.485 | 308.906 | 3.474 | 0 |
| prior | cpu | 512 | forward | PASS | 2029.331 | 2055.051 | 0.494 | 0 |
| prior | cpu | 512 | end_to_end | PASS | 2048.723 | 2078.331 | 0.488 | 0 |
| student | cpu | 512 | forward | PASS | 1735.502 | 1791.252 | 0.578 | 864752 |
| student | cpu | 512 | end_to_end | PASS | 1732.157 | 1808.618 | 0.574 | 864752 |
| dcp | cuda | 256 | forward | PASS | 0.153 | 0.167 | 6461.452 | 0 |
| dcp | cuda | 256 | end_to_end | PASS | 3.541 | 6.337 | 238.736 | 0 |
| prior | cuda | 256 | forward | PASS | 3.314 | 5.800 | 263.860 | 0 |
| prior | cuda | 256 | end_to_end | PASS | 6.297 | 6.417 | 156.740 | 0 |
| student | cuda | 256 | forward | PASS | 9.411 | 9.650 | 105.086 | 864752 |
| student | cuda | 256 | end_to_end | PASS | 12.842 | 13.081 | 77.499 | 864752 |
| dcp | cuda | 512 | forward | PASS | 0.323 | 0.453 | 2890.565 | 0 |
| dcp | cuda | 512 | end_to_end | PASS | 8.375 | 8.667 | 117.282 | 0 |
| prior | cuda | 512 | forward | PASS | 2.845 | 3.597 | 326.290 | 0 |
| prior | cuda | 512 | end_to_end | PASS | 10.653 | 10.794 | 93.972 | 0 |
| student | cuda | 512 | forward | PASS | 26.385 | 26.567 | 37.905 | 864752 |
| student | cuda | 512 | end_to_end | PASS | 34.733 | 35.393 | 28.648 | 864752 |

显存与原始逐次延迟见 `performance.json`。CPU 显存为 N/A；CUDA 缺失为 NOT RUN。MACs 仅统计 Conv/Linear，不能冒充完整 FLOPs。

`student_architecture_only` 使用未训练的相同网络，仅反映结构计算成本，其正确性仍是 NOT EVALUATED。设备不同的数字不能直接证明算法优劣。
