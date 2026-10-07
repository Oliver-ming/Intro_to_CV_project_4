# Keep upstream checkouts out of experiment code

实验逻辑放在 `src/`，按阶段拆开：`data`、`init`、`reconstruct`、`generate`、`eval`、`run`。上游仓库只放在 `third_party/`，固定版本，不在里面写本项目的实验流程。`configs/` 放实验配置。`data/`、`weights/`、`outputs/` 不进 git。Python 环境是 `../env_cv`。

这样查找时，初始化、重建、生成和评测不会混在克隆下来的官方仓库里。

## Considered Options

- `src/` 按阶段拆分，上游只读地放在 `third_party/`（采用）
- 直接在各个上游仓库里改实验代码
