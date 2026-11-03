# 候选校正阶段提交与复现

本次提交同时包含冻结候选动态融合的不足和联合校正的改善，不只发布好成绩。最新主报告为 ../joint_correction_review_20261102.md。完整源码、两个配置及独立实验工具纳入 Git；两轮全部种子、三种条件逐样本指标、候选偏差/共同错误/oracle、完整成本、协议和训练记录位于 experiments/candidate_correction_delivery_20261102。

本次未自动提交或推送。已按 manifest 精确暂存，可先查看 git diff --cached --stat，然后执行：

    git commit -m "feat(research): 整合受限候选联合校正与完整对照证据" -m "归档冻结候选动态融合、联合校正及三种子同训练机会对照；保留候选独立误差、空间权重干预、完整推理成本与域外失败记录。开发集结果有竞争力，但尚无速度优势、独立语义物理贡献或新确认结论。"

随后用户自行执行 git push origin v1-nocolab。本地归档已离线核验；推送不影响云端旧实验恢复权重。

## 已完成结果的离线检查

    python scripts/joint_review/verify_delivery.py

该命令核对 manifest SHA256，并从两轮各三种条件的 CSV 重新计算各方法 MAE，不加载数据或权重。归档原始 manifest 引用的未发布大文件仍位于原 /mnt，不能因不在 Git 中就改写其历史哈希。

## 云端运行与恢复

联合实验的原工作树：/mnt/CrossModal-Defog-OVOD-joint-correction-20261102。
冻结候选实验工作树：/mnt/CrossModal-Defog-OVOD-matched-gate-20261102。
预训练来源：/mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102/formal_amp。
各工作树 experiments 下保留 best/latest、portable_weights、源码快照和原协议；Git 归档仅保存清单、哈希及数值记录。

联合训练原命令为 python -u scripts/joint_correction/run.py --stage all；运行监控为 python scripts/joint_correction/status.py --watch。在已完成目录重启不会提供新训练，重新实验须新目录、新协议；冻结恢复必须使用匹配源码快照，不跳过源码哈希。配置中的 /mnt 路径是已记录的本机路径，迁移实例前须核验、在新配置副本中调整，不修改冻结配置。

RGB-only 推理入口：scripts/joint_correction/predict.py；需要原 portable_weights/joint_dynamic_seed42.pth 或其它已登记种子。输入应符合训练生成过程：linear_srgb 是线性域加雾后转回显示 RGB，网络并非直接吃线性图。display_rgb 压力测试改变的是加雾机制，不能简单用逆 gamma 当作补救。脚本不会自动识别传感器编码，不可把任意 PNG 当作同域输入或用真实 t/A 校正。

GitHub 同步到云端继续使用 github_sync.md 的“新工作树接收”办法。仓库里没有权重，不能用 pull 覆盖原实验目录。此次为提交整理另建 /mnt/CrossModal-Defog-OVOD-delivery-20261102，原两轮工作树不变。

## 本次精简边界

源码和最终配置保留全部，既有导入依赖位于同一提交或上一提交。证据仅保留一份精选归档，不重复加入本地 cloud_evidence、source 快照、备份或同步回执。小型原始日志保留以核对命令和训练过程，权重/数据/缓存留网盘。后续方案不会因本次开发结果自动进入新确认或真实来源验证。

后续诊断顺序与预算假设见 [性能与域偏移后续计划](../candidate_correction_next_steps_20261102.md)，仅为计划，未启动新训练。
