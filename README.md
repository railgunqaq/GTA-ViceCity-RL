# reVC RC Helicopter RL

基于 reVC 的 GTA Vice City 遥控直升机强化学习实验，包含 Windows 原生桥接、Python 环境、SAC/DQN、几何查询与路线诊断。

**当前是研究原型。** 已验证规划器与 PD 控制器完成第一层一次无损投放；尚未证明 SAC 学会投放或完成四目标任务。原生桥接为 v7，环境为 rc-heli-v5，SAC checkpoint 格式为 v4，观测 76 维、动作 5 维。

此目录是经过筛选的源码发布副本。完整游戏、游戏资源、预编译程序、SDK 二进制、模型权重、原始训练日志和个人配置均不在仓库中。实机运行需要自行准备本地游戏数据和匹配的原生构建。上游来源与许可声明见 [NOTICE.md](NOTICE.md)。

## 目录

```text
ml/                         当前训练、诊断和分析代码
ml/tests/                   不连接游戏的离线回归测试
reVC-src/src/ml/             原生共享内存控制与遥测桥接
reVC-src/                   原生源码和构建配置；librw 源码随附
audit/                      精简验证摘要、路线提示和下一阶段评估
docs/                       构建、证据边界及发布说明
tools/check_publish.py      发布文件检查
requirements.txt            Python 核心依赖
requirements-plot.txt       可选绘图依赖
```

## 安装与离线检查

在仓库根目录，用自己的 Python 环境执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s ml/tests -v
```

核心依赖为 NumPy 和 PyTorch；绘图另需 `requirements-plot.txt`。依赖文件未锁定版本，不代表已经验证所有组合。本次发布副本的测试环境与结果记录在 [发布检查记录](docs/RELEASE_VALIDATION.json)。原生环境目前面向 Windows；离线测试通过不代表其他平台可以运行原生桥接。

## 实机使用

先按 [原生构建说明](docs/BUILD.md) 准备匹配的可执行文件、依赖和本地游戏数据。下列命令会启动游戏，应在准备完毕后手动执行：

```powershell
$env:REVC_GAME_DIR = 'C:\VC'
python -B ml/smoke_contract.py --output audit/local/smoke.json
python -B ml/smoke_training.py --output audit/local/smoke-training
```

`REVC_GAME_DIR` 统一设置训练和诊断入口的默认游戏目录；未设置时沿用 `C:\VC`。提供 `--game-dir` 的入口可单独覆盖。每次使用新的输出目录。

完整接口、动作与奖励契约、受限实验命令见 [ml/README.md](ml/README.md)。旧环境模型不能直接用于当前环境，也不随本仓库发布。

## 已有证据与下一步

- [墙体与通路验证](audit/2026-11-06/WALL_NAVIGATION_FIX.md)：原生静态碰撞检查、一次无损进楼及真实放置事件。
- [下一阶段评估](audit/2026-11-13/NEXT_STAGE_ASSESSMENT.md)：当前直线距离停滞判据会在成功轨迹第 728 步提前结束，而完整单点轨迹需要 2775 步；训练上限也只有 1500 步。首枚投放用掉约 64% 任务倒计时。
- 推荐先做航点条件化、修正进度判据、复现参考控制器及验证回程耗时，再进行示范初始化和小预算分段学习。这些后续改进尚未实现。
- 精简证据的来源、删减和可复核边界见 [EVIDENCE.md](docs/EVIDENCE.md)。摘要不能代替完整原始轨迹的独立复算。

## 发布

本地仓库已经准备为 `main` 分支，未创建远程仓库或推送。提交前运行：

```powershell
python -B tools/check_publish.py
```

创建提交、关联远程和推送的步骤见 [PUBLISH.md](docs/PUBLISH.md)。只上传这个发布目录；原始工作区含完整游戏备份和大量运行产物。
