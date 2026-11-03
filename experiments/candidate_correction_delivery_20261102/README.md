# 受限候选校正：精选提交归档

本目录保存两轮实验及离线复核的完整数值证据，保留成功和失败条件，排除权重、缓存、预计算数组、重复源码快照、传输清单和机器启动文件。源码位于项目 src/、scripts/，配置位于 configs/，研究报告位于 docs/。

- matched_fusion：冻结候选＋动态融合，以及相同训练机会的 RGB 控制。校正动态 MAE 0.012211，落后 RGB16 的 0.009582。
- matched_review：同输入 RGB 预测投影到候选区间的离线诊断，仅解释范围和学习差距，不是部署模型。
- joint_correction：两个新模式、三个种子，联合动态 MAE 0.007261，参数相近但推理较慢；保留未见 A 和显示域失败结果。
- joint_review：逐样本重算、候选独立误差、共同错误、oracle 下界、分层及权重干预复核。

全部场景均为已暴露开发参考，不能作为新确认。manifest.json 单独记录 Git 发布文件及原云端路径，原始协议、源码/权重哈希清单不重写。完整恢复权重仍在原 /mnt 工作树，不进入本目录。

从项目根目录运行 python scripts/joint_review/verify_delivery.py，可离线核验发布字节及两轮三条件 MAE。无需 GPU、数据集、模型权重或第三方 Python 包。具体运行和恢复边界见 docs/guides/candidate_correction_submission_20261102.md。
