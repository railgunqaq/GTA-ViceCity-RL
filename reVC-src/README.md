# reVC 原生环境源码

本目录保存 RC 直升机强化学习项目使用的 reVC 源码及 Win32 训练桥接。原项目说明原文保存在 [README.upstream.md](README.upstream.md)，包括原作者、历史构建方式及 License 段；其中的旧下载地址和构建服务状态未在本次整理中验证。

## 当前范围

- [src/ml/MlBridge.h](src/ml/MlBridge.h) 实现协议 v7：共享内存与互斥同步、固定 10ms 原生帧步进、控制输入、任务事件和只读静态几何查询。
- 原生调用接入 `src/core/`、`src/control/` 等现有游戏循环；Python 环境与学习器位于相邻的 [ml/](../ml/README.md)，使用 rc-heli-v5 契约。
- 保留本地已有的 `vendor/librw` 源码及其独立许可证，保留 `vendor/milessdk/include/mss.h` 兼容头。
- 保留渲染器直接包含的 `src/extras/shaders/obj/*.inc` 文本数组，以及 Windows 资源文件引用的图标。这些文件属于源码构建输入，不能与普通 `obj/` 编译输出一起排除。

桥接当前依赖 Windows API。上游引擎的其他平台构建选项不代表本项目的训练桥接已在那些平台验证。

## 构建与运行

见 [原生构建说明](../docs/BUILD.md)。本发布副本排除了游戏资源、预编译工具、第三方 DLL/LIB、已有构建目录及可执行文件，需自行配置外部依赖和本地游戏安装。本次整理没有对这个发布副本执行原生构建或游戏测试。

当前依赖按普通源码目录保存，不是 Git 子模块；旧子模块声明仅作来源记录，见 [upstream.gitmodules.txt](../docs/upstream.gitmodules.txt)。原始源码副本不包含上游 Git 历史，本仓库不能据此提供准确的上游提交号。

## 来源与声明

上游 README 的 License 段没有授予统一的标准开源许可证。本仓库保留该声明，没有为整个引擎或所有依赖新增 MIT/Apache 许可。各组件的说明与已保留许可证见 [NOTICE.md](../NOTICE.md)。
