# 2026-11-03 优化精选归档

从 GitHub 提交 ed4cc48 在 /mnt 独立工作树进行。源码在 src/models/haze_density 和 scripts/optimization_20261103，配置在 configs；昨日简报和当日两份报告在 docs。

- latency_run03：三模型公平 CUDA 图计时、三轮重复、512 旧开发样本输出差零、输入拒绝及输出所有权检查。
- separable_check：严格等价检查失败，记录原门槛和数值差。
- approximate_quality：近似执行的独立质量门槛、完整计时、三条件逐样本结果。不是数值等价；仅 seed42 开发检查，不能冒充新确认。
- airlight_pilot：三模型三个种子相同六轮追加训练；保留名义域退步、大气光改善、显示域压力和每候选诊断。旧训练标定固定，用于隔离 A 覆盖。
- audit、budget、各日志：独立重算、48 份旧权重保护、训练预算及已修复的构建/导出问题。原训练源码保留，导出通过独立 finish_export.py 恢复。

联合模型相对旧 RGB16 的完整 GPU 推理差距尚未消除。近似版本 256 约 2.78 ms，对照 RGB 约 1.96 ms，仍慢 41.5%。随机 A 增强使 unseen_A MAE 0.079443→0.028371，nominal 0.007261→0.032748，是取舍，不是普适解决方案。

离线核验：python scripts/optimization_20261103/verify_delivery.py。
云端统一入口：python scripts/optimization_20261103/run.py；已完成阶段直接保留。监控：python scripts/optimization_20261103/status.py --watch。
本轮未改根 .gitignore；仅按 manifest 精确强制纳入必要证据，今后新输出仍默认忽略。

best/latest、portable_weights、图片和 NPY 原始图/预测、重复 source 快照留在 /mnt/CrossModal-Defog-OVOD-optimize-20261103/experiments/optimization_20261103，不进入 Git。归档映射与 Git 字节哈希独立保存，不修改历史协议。完整复制冻结源码时，需匹配对应原 worktree 的 HEAD 和协议；禁止在冻结目录追加方案。
