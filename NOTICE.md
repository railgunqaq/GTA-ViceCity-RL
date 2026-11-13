# 来源与许可说明

本文件记录发布副本中可直接检查的来源和现有声明，不对全部代码授予新许可证，也不替代各组件自己的文件。

## reVC / re3

`reVC-src` 来源于原有的 reVC 工作目录。上游说明将项目标识为 GTAmodding/re3 的 miami / reVC 分支，介绍其反向工程来源、作者和历史。本副本将该说明原文保存在 [README.upstream.md](reVC-src/README.upstream.md)，并保留已有源码注释。

上游 README 的 License 段表示作者不认为自己处于可以为该代码授予许可证的位置，并描述教育、文档和 modding 用途、不鼓励盗版或商业用途、要求保留署名等立场。原副本没有统一的根 LICENSE，CMake 中也保留了 missing license 注释。因此本仓库没有把整个引擎或本项目整体标为 MIT、Apache 或其他标准许可证；上游 README 中的表述也不在此被改写为新的许可授权。

本次导出的工作目录没有上游 Git 元数据，无法从该副本确认准确的上游提交号或恢复原始提交历史。历史构建记录中的二进制哈希不等于上游源码提交号。

## librw 与附带组件

`reVC-src/vendor/librw` 按普通 vendored 源码目录保留，现有文件注明 Copyright (c) 2014 aap，并包含独立的 [MIT License](reVC-src/vendor/librw/LICENSE)。该许可证的适用范围不能扩展到整个 reVC 引擎或其他未以该许可证发布的文件。

该目录中另保留：

- [Dear ImGui 许可证](reVC-src/vendor/librw/skeleton/imgui/LICENSE_imgui.txt)。
- [ImGuizmo 许可证](reVC-src/vendor/librw/skeleton/imgui/LICENSE_imguizmo.txt)。
- 其他文件内部已有的作者、版权及许可注释。

## Miles 兼容声明与外部依赖

`reVC-src/vendor/milessdk/include/mss.h` 的文件注释将其描述为 re3 使用的 fake 兼容头，并给出原兼容项目来源。这里保留该头及其注释，不将其描述为完整官方 SDK，也没有新增许可授权。其预编译导入库和运行时 DLL 不在发布副本中。

旧子模块 URL 声明作为 [upstream.gitmodules.txt](docs/upstream.gitmodules.txt) 保留，仅用于来源记录；当前仓库没有相应的 Git 子模块条目。未打包的 OpenAL、mpg123、libsndfile、ogg、opus、opusfile 等依赖应由使用者按各自的版本和许可获取。

## 本项目新增内容与运行资源

本项目的 Python 训练/诊断代码、原生桥接改动和实验报告没有在本次整理中被赋予新的仓库级开源许可证。后续若选择许可证，应明确其覆盖范围，并保留上游及第三方文件的独立声明。

发布副本不包含完整游戏、游戏安装目录、任务 SCM、游戏数据包、预编译引擎或第三方运行时库。源码构建所引用的图标和着色器文本数组按原路径保留，不因这次整理而新增许可声明。实机运行依赖自行准备的游戏资源，具体构建范围见 [BUILD.md](docs/BUILD.md)。
