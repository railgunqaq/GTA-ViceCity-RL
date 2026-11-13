# 上传 GitHub

这个目录是独立的、尚无提交历史的本地 Git 仓库。分支为 `main`，发布文件已加入暂存区；没有配置远程或执行推送。原工作区和游戏安装不属于此仓库。

## 本地复核

```powershell
git status --short
git diff --cached --stat
python -B tools/check_publish.py
python -B -m unittest discover -s ml/tests -v
```

检查范围包含暂存清单、常见凭据格式、个人目录、游戏数据及模型/构建产物。它是静态筛查，不是凭据安全或许可审查的保证。原生来源和独立依赖声明见 [NOTICE.md](../NOTICE.md)，导出范围见 [EVIDENCE.md](EVIDENCE.md)。本次没有为整个项目新增统一开源许可证。

## 创建提交与远程

在 GitHub 创建空仓库，自行选择仓库名和可见性；不要同时生成另一份 README 或 LICENSE。然后在本目录执行，替换示例地址：

```powershell
git commit -m "Prepare RC helicopter RL source snapshot"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

提交需要本机已配置 Git 作者身份。通过系统凭据管理器或 SSH 完成身份验证，勿把 token 写进 remote URL、脚本或文档。若远程已经有提交，应先检查远程历史，不要直接强制推送。

随附 ZIP 是同一发布文件集的便携副本，不含 `.git`。若使用 ZIP，先解压，再自行初始化仓库；不要把 ZIP 当作唯一仓库内容。

## 后续更新

发布目录是一次源码快照。继续开发时应明确选用哪个目录；原工作区的新修改不会自动同步到这里。发布副本对游戏目录配置做了 `REVC_GAME_DIR` 适配，原训练工作区没有改动。

新实验结果默认写入被忽略的 `ml/runs/` 或 `audit/local/`。只挑选摘要加入仓库，不要强制加入完整游戏、旧备份、个人配置、检查点或原始大日志。变更后重新运行文件检查和离线测试。
