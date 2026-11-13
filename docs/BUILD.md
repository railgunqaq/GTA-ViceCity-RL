# 原生环境构建说明

本说明针对发布副本的源码布局。**整理阶段未对此副本执行原生构建或实机测试。** 历史工作目录曾构建 Windows x86、librw D3D9、MSS 配置；历史成功记录不能替代这个干净副本的依赖恢复和重新验证。

## 发布副本包含的构建输入

- `reVC-src/src/`、`cmake/`、Premake/CMake/Conan 配置和版本信息生成脚本。
- `reVC-src/vendor/librw/` 源码和原有许可证；不是预编译 `rw.lib`。
- `reVC-src/vendor/milessdk/include/mss.h`。文件开头将其描述为 re3 使用的 fake 兼容头；这不是完整的官方 Miles SDK，且不包含导入库或运行时 DLL。
- 渲染器所需的 `src/extras/shaders/obj/*.inc`、相应着色器源码及 `win.rc` 引用的图标。常规编译使用现有 `.inc`；重新生成 HLSL 着色器还需配置编译脚本要求的 `fxc.exe` / `DXSDK_DIR`。

本副本不包含 `gamefiles/`、游戏安装、任务 SCM、预编译 Premake、游戏/第三方 EXE、DLL、LIB、PDB、已有 Visual Studio 工程生成目录和历史训练产物。上游的 `utils/gxt` 文本资源、旧发布工作流与部分 librw 演示资产也未打包。

## 外部依赖

### Windows x86 / D3D9 / MSS 参考配置

1. 配置 Visual Studio C++ Build Tools、Windows SDK 和兼容的 Premake 5，让 `premake5` 与 `MSBuild.exe` 可在 Developer PowerShell 中调用。本次没有验证 Premake 发行版本。
2. 提供与 x86 目标匹配的 `mss32.lib`，按现有 Premake 配置放到 `reVC-src/vendor/milessdk/lib/mss32.lib`，或自行调整本地库搜索目录。仅有本仓库的 `mss.h` 无法完成链接。
3. 编译需可用的 Windows/DirectX 头文件和导入库。参考配置链接 D3D9；RW34/D3D8 等其他平台条目另有 SDK 要求，不是本项目的参考训练配置。
4. 实机运行需自行准备本地 GTA Vice City 游戏资源及该音频配置要求的运行时文件。此源码副本不会下载、生成或补齐游戏资产，也不附带完整游戏安装。

不要将新增的本地 SDK、导入库、DLL 或游戏文件提交到仓库。

### 其他配置的限制

- OpenAL 配置需外部提供 OpenAL、mpg123，并按所选 Premake/CMake 选项提供 libsndfile 等依赖。当前副本没有这些原 vendor 目录中的二进制和头文件。
- GLFW/OpenGL 配置还需相应图形依赖。Windows Premake 使用 `--glfwdir32` / `--glfwdir64` 指定路径。
- 上游 `.gitmodules` 曾列出 ogg、opus、opusfile 和 librw。它只作为 [来源记录](upstream.gitmodules.txt) 保存：本仓库没有对应 Git 子模块条目，`git submodule update` 不会恢复这些库。
- Opus 默认关闭；启用 `--with-opus` 或对应 CMake 选项前，需要独立准备 ogg、opus、opusfile 的头文件与匹配库。
- librw 示例资源未完整打包。走 CMake 路线时可设置 `LIBRW_TOOLS=OFF`，避免将未验证的演示工具视为本项目构建目标。

## 生成 Visual Studio 工程并编译

先恢复上述外部依赖，再从仓库根目录打开 Developer PowerShell。以下命令展示原工程生成流程和历史已使用的编译器/SDK 组合；工具集与 SDK 版本需按本机安装情况选择。

```powershell
Set-Location reVC-src

# 避免 Premake 生成自动覆盖本地游戏可执行文件的 post-build 操作。
Remove-Item Env:GTA_VC_RE_DIR -ErrorAction SilentlyContinue

# Premake 需由使用者安装；本仓库不附带 premake5.exe。
premake5 vs2019 --with-librw

# 历史工作目录使用 VS 2022 Build Tools、v143、Windows SDK 10.0.26100.0。
MSBuild.exe .\build\reVC.sln /p:Configuration=Release /p:Platform=win-x86-librw_d3d9-mss /p:PlatformToolset=v143 /p:WindowsTargetPlatformVersion=10.0.26100.0 /m:2 /nologo /v:minimal
```

预期输出位置为 `reVC-src/bin/win-x86-librw_d3d9-mss/Release/reVC.exe`。Premake 会同时构建 vendored librw；`printHash.bat` 生成的 `src/extras/GitSHA1.cpp` 是本地构建产物。新仓库的 Git SHA 标识当前仓库版本，不代表恢复了原始上游版本。

`premake-vs2015.cmd`、`premake-vs2017.cmd`、`premake-vs2019.cmd` 保留为上游入口；它们同样要求外部 `premake5`，且不是对所有版本组合的验证承诺。

## CMake / Conan

保留的配置可用于后续适配，不能视为已经验证的替代构建流程。CMake 默认音频配置为 OAL，使用 `find_package` 查找外部依赖，当前 `src/CMakeLists.txt` 还会无条件查找 mpg123。Conan 文件保留上游接口和依赖声明，本次没有完成依赖解析或验证其兼容版本。

由于本副本排除了 `gamefiles/`，不要将上游的安装/CPack 流程当成完整可运行游戏包生成器。构建引擎与准备本地运行数据是两个独立步骤。

## 构建后的验证

编译成功后再手动部署到独立的本地测试安装；替换前结束自己的游戏实例并备份旧可执行文件。接着按 [ML 文档](../ml/README.md) 核对桥接版本、任务脚本和环境契约，执行接口检查后再运行训练。新旧模型、观测、奖励与原生桥接需要保持一致，不能仅凭动作/观测维数相同就混用。

记录本次实际工具版本、构建命令、源码提交、二进制哈希和实机结果。只有这些新结果才能证明该发布副本在相应机器上的可复现状态。
