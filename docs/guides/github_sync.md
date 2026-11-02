# 本次提交与云端网盘同步

后续源码、配置、报告编辑和实验在云实例 `/mnt` 网盘完成；本地用于查看、同步及 Git 交付。用户负责最后推送，本次未执行 push。

## 这次选择了什么

本次包含语义辅助诊断、物理可行性复核、CPU 内容表校正、V100 受限神经校正的独立源码/配置/报告，以及支撑结论的协议、数据清单、逐例成功和失败指标、训练记录、预算与权重哈希。不是只选择通过的实验。精确发布范围见 `experiments/research_delivery_20261102.json`。

重复的 pilot_snapshot/development_snapshot、中间 source_amp、个人 SSH 连接助手、机器审计备份、权重、图片、大型缓存、ZIP 均未选入 Git，仍留在本地或 `/mnt`；原始实验完整哈希清单允许引用这些未发布文件。Git 发布清单单独记录实际 Git blob 的 SHA256，不修改原始实验清单。

推送前在本地查看 `git show --stat HEAD`，最后执行：

```powershell
git push origin v1-nocolab
```

本次没有训练或新增测试场景。新确认集已暴露，后续不能再次当作未使用的确认集。

## 推送后同步到云实例

原项目下有 Git 忽略的权重、恢复状态和云端独有结果，直接 pull 可能覆盖后来被纳入 Git 的同名忽略文件。因此使用新工作树接收代码，原项目的 HEAD、未提交内容和实验文件保持原状。

云端已有一份本次准备的同步工具，可在首次拉取代码前使用：

```bash
python /mnt/CrossModal-Defog-OVOD/experiments/github_delivery_20261102/prepared/scripts/utils/sync_github_worktree.py \n  --repository /mnt/CrossModal-Defog-OVOD \n  --branch v1-nocolab \n  --destination /mnt/CrossModal-Defog-OVOD-sync-20261102 \n  --expected-commit <本次提交的完整40位哈希> --apply
```

完整提交哈希从本地 `git rev-parse HEAD` 获取。省略 `--apply` 只预览并检查远端分支是否等于预期提交。目标目录必须尚不存在，所有路径解析后必须位于 `/mnt`；工具拒绝与旧项目重叠的目录，不 reset、不 clean、不覆盖旧工作树。若远端尚未推送或已前进，直接退出。

同步成功后在新工作树继续工作：

```bash
cd /mnt/CrossModal-Defog-OVOD-sync-20261102
git switch -c codex/next-research
```

已有结果仍在 `/mnt/CrossModal-Defog-OVOD/experiments/`，不复制、不删除。新实验使用新的 `/mnt` 输出目录；不要向冻结的 formal_amp 或 source_formal 原地写入新方案。

## 权重和复现

V100 完整恢复状态：`/mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102/formal_amp/`。各模式/种子的 best/latest 保留原位。全部九份小型推理权重：同实验下 `all_seed_weights/`；总体标定：`cost_assets/calibration.json`。Git 仅发布它们的清单和哈希。

在新的云端工作树中可用原网盘权重执行独立文件推理：

```bash
python scripts/learned_correction/predict.py \n  --image /mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102/cost_assets/input_rgb.png \n  --weights /mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102/all_seed_weights/corrector16_seed42_inference.pth \n  --calibration /mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102/cost_assets/calibration.json \n  --output /mnt/CrossModal-Defog-OVOD-new-runs/inference_NEW --device cpu
```

输入示例和模型都留在原网盘，输出目录应换成新的唯一名字。Git 发布的正式指标在 `experiments/learned_correction_20261102/final_evidence/`；原始云端运行目录则没有 final_evidence 这一层。这个区别不改变指标，只是本地取回时的归档结构。

冻结运行恢复必须使用原始 `source_formal` 快照及匹配哈希；当前代码新增归档/分析工具后，不能强行绕过源码集合哈希检查恢复旧目录。新源码从零复现请使用新目录。语义试验还需要额外的冻结 SegFormer 权重，既有结果不需要重跑。
