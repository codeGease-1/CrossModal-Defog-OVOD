# v2 可复现实验记录

结论：最小空间门控已实现并完成三个种子验证，但趋于单暗通道，当前不建议进入最终主流程。中文解释见 [报告](../../docs/v2动态加权验证与v15继续核验报告.md)。

本目录为本次完整结果，不覆盖 `v15_20261101`。训练和参考检查点均保留，未自动提交 Git。云端基准提交 `ce19129`；本地基准较早，不能将本地未跟踪的 v1.5 文件误认作本轮新增研究结果。

## 环境与输入

- 云端仓库 `/mnt/CrossModal-Defog-OVOD`；V100 16GB，Python3.12，torch2.4.1+cu124；具体包见 `pip_freeze.txt` 和 `protocol.json`。
- 原数据压缩包 `/mnt/RSHaze+.zip`：按 ZIP 成员读取，不需全部解压；原始数据不入 Git。
- 依赖原 `experiments/v15_20261101/{protocol,calibration,scene_manifest,external_manifest}.json` 和 `audit/archive_audit.json`，以及原脚本/指标实现。文件或源哈希不一致会拒绝继续已有实验，应新建目录。
- `scene_manifest.json` 保存受控原始场景种子及分组；`external_manifest.json` 保存新成员与解码哈希。所有 RSHaze 真实来源暂归同一地理组，只作外部检查。

## 运行命令

在 Linux 仓库根目录执行。以下主阶段接受新目录；不要为绕过保护而修改已有 `protocol.json`。

```bash
bash scripts/v2/reproduce.sh experiments/v2_NEW
```

`prepare → calibrate → diagnose → train → freeze → development → test → confirmation → external` 是主流程。先看 oracle 诊断再训练：若多数关键错误同向不可修正，代码只允许小试验。`test` 是历史诊断集，`confirmation` 才是本轮冻结后的新受控确认集。不要用二者反复调参。

本次完整扩展试验采用下列顺序。辅助脚本使用固定目录 `experiments/v2_20261101`；它们通常在结果/完成标记存在时直接返回。需要从零复现时，在新的仓库副本放置原 v1.5 依赖并从空的同名目录开始，保留原副本历史，切勿删除已有结果。辅助脚本的路径限制不影响主流程新目录运行。

```bash
python scripts/v2/test_v2.py
python scripts/v2/test_solver.py
python scripts/v2/run.py --stage prepare
python scripts/v2/run.py --stage calibrate
python scripts/v2/run.py --stage diagnose
python scripts/v2/smoke_train.py
python scripts/v2/run.py --stage train
# watch_weights.py 只能在训练期间另外运行；历史观测并不完整。
python scripts/v2/ranking_ablation.py
python scripts/v2/run.py --stage freeze
python scripts/v2/run.py --stage development
# 加强 B 的求解由开发集边界收敛诊断触发，只用训练集。
python scripts/v2/global_optimum.py
python scripts/v2/run.py --stage test
python scripts/v2/run.py --stage confirmation
python scripts/v2/run.py --stage external
python scripts/v2/supplement.py
python scripts/v2/exact_evaluate.py
python scripts/v2/v15_checks.py
# 显示域兼容标定单独冻结、单独新样本，不改变主 v2。
python scripts/v2/v15_display_compat.py
python scripts/v2/plot_failures.py
python scripts/v2/verify_resume.py
python scripts/v2/finalize.py
```

源文件中不会包含 SSH 私钥或访问凭据。环境需要 torch、numpy、Pillow、scipy、matplotlib，完整版本按环境记录恢复。主训练同一命令支持从 `latest.pth` 恢复：保存模型、优化器、epoch、shuffle/torch/CUDA RNG；完成后重跑会直接返回。预计算特征驻 CPU 内存，峰值需数 GB；不持久化庞大训练特征。校准缓存未打包，缺失时可再生成。

## 结果索引

| 文件/目录 | 用途 |
|---|---|
| `protocol.json`, `completion.json`, `artifact_sha256.json` | 运行前协议、补充脚本哈希与环境、最终文件完整性 |
| `calibration.json`, `frozen.json` | 训练集候选图标定、开发选模冻结记录 |
| `oracle_development.json`, `oracle_per_case.csv` | 理论凸融合能力边界与地物分项 |
| `global_seed*/`, `dynamic_seed*/` | 三种子 best/latest 检查点、优化器与历史 |
| `global_optimum/fit.json` | 精确固定权重，仅训练集求解 |
| `development/`, `test/`, `confirmation/`, `external/` | summary、逐例 CSV、离线 oracle、失败清单和图 |
| `confirmation/figures/FAIL_*.png` | 实际最差例候选图和塌缩权重，统一 [0,1] |
| `original_spectral/` | 新 S 成员原始合成图、逐通道重建核验、次要终点 |
| `supplement.json` | 3种子配对 bootstrap、耗时、权重和高亮/低纹理代理指标 |
| `ranking_ablation/` | 单种子排序损失消融、确认结果与参数差 |
| `v15_continuation/` | 成像域配对试验、单调表 bootstrap、支持范围检查 |
| `v15_display_compat/` | 显示域标定、另一批确认场景/原始S、改善与退步证据 |
| `budget.json` | 训练+标定计时；非总运行时间或云费用 |
| `unit_tests.log`, `resume_check.log`, 其他 `*.log` | 核验及原始运行日志；恢复检查成功时日志为空 |
| `resume_fixture.json` | 实际训练器 CPU 微型数据在第一轮保存后中断，恢复后参数与连续训练逐项一致；不模拟 CUDA 硬件故障 |

早期 `smoke.log` 有文件名遮蔽 Python 标准库造成的失败，修正为 `smoke_train.py` 后成功，见 `smoke_retry.log`；观测器并发读写曾失败，重试记录在 `weight_observer_retry.log`。错误日志保留，不影响最终检查点和指标。权重轨迹仅覆盖后期，最终塌缩判断来自冻结模型的完整逐样本指标。

各原始错误样本保持固定公共色标；`oracle_OFFLINE_ONLY` 只可离线解释能力边界。所有模型只接收由 RGB 计算的输入。真实图不是按逐图真值挑选标定；未知成像域仍是阻碍。

Git allowlist 保存配置、证据、best/latest 权重及有限图像。完整下载目录还包含额外统一色标图；校准缓存可由代码再生。不得因目录中有真实参考图的追加合成结果，就宣称有可靠真实大气浓度标签。
