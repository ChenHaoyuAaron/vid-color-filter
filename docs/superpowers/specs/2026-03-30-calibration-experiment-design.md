# Calibration 实验设计：阈值标定与可视化审核

## 目标

对上千对视频跑 Temporal S-CIELAB 管线，通过网格搜索 + 人工标注边界案例，找到 `global_threshold` 和 `local_threshold` 的最优组合。

## 约束

- 管线在**远程服务器**（有 GPU）上运行
- 生成的可视化产物下载到**本地**浏览和标注
- HTML 报告必须完全自包含（相对路径引用 PNG，无服务器依赖）
- 标注通过 HTML 页面内嵌 JS 按钮完成，用 `localStorage` 暂存，支持导出 JSON

## 流程

```
服务器端：
1. 全量跑管线 → scores.jsonl
2. 生成全量分布图 → distribution.html（内嵌 base64 PNG）
3. 网格搜索统计 → grid_search_preview.html（各阈值组合下 pass/fail 数量）
4. 筛选边界案例（评分接近候选阈值的视频对）
5. 为边界案例生成可视化 PNG + HTML 报告
6. 打包 calibration_output/ 目录

本地端：
7. 下载 calibration_output/
8. 打开 index.html，浏览边界案例报告
9. 在报告页面中点 pass/fail 按钮标注
10. 导出 annotations.json
11. 上传 annotations.json 到服务器

服务器端：
12. 加载 annotations.json，对每组阈值计算 precision/recall/F1
13. 生成最终 grid_search_results.html（F1 heatmap）
```

## 输出目录结构

```
calibration_output/
├── scores.jsonl                 # 全量评分结果
├── distribution.html            # 全量分布图（自包含，base64 PNG）
├── grid_search_preview.html     # 网格搜索预览（标注前，仅 pass/fail 数量）
├── grid_search_results.html     # 网格搜索结果（标注后，F1 heatmap）
├── boundary_cases.json          # 边界案例列表及元数据
├── annotations.json             # 人工标注结果（标注后生成）
├── reports/
│   ├── index.html               # 索引页（按分数排序，链接各报告，内嵌标注 UI）
│   ├── {video_pair_id}/
│   │   ├── report.html          # 该视频对的可视化报告
│   │   ├── src_frame_{i}.png    # 源帧（3-5 个代表帧）
│   │   ├── edit_frame_{i}.png   # 编辑帧
│   │   ├── heatmap_{i}.png      # S-CIELAB ΔE heatmap（colormap 叠加源帧）
│   │   ├── mask_{i}.png         # 自适应 mask overlay
│   │   ├── median_map.png       # 时域中值 ΔE 空间分布
│   │   └── iqr_map.png          # 时域 IQR 空间分布
```

## 模块设计

### 模块 1：可视化生成器 (`src/vid_color_filter/gpu/visualizer.py`)

在 GPU 管线中生成可视化所需的原始数据，存为 PNG。

**输入**：管线中间产物（S-CIELAB 滤波后 Lab、ΔE map、mask、temporal median/iqr map、原始帧）

**生成的可视化**：

1. **帧级对比**：
   - 源帧、编辑帧直接存为 PNG
   - 代表帧选取：按 per-frame mean ΔE 排序，取最高、最低、中位数帧，再加 2 个均匀分布帧，共 5 帧

2. **ΔE Heatmap**：
   - 用 matplotlib `viridis` colormap 将 ΔE map 渲染为彩色图
   - 半透明叠加在源帧上（alpha=0.6）
   - Colorbar 标注 ΔE 值范围
   - Mask 区域用半透明灰色覆盖（表示排除区域）

3. **Mask Overlay**：
   - 源帧上叠加半透明红色 mask（alpha=0.4）
   - 标注 mask coverage 比例

4. **Temporal Median Map**：
   - 跨帧聚合后的中值 ΔE 空间分布，viridis colormap
   - 与单帧 heatmap 共享 colorbar 范围以便对比

5. **Temporal IQR Map**：
   - 帧间波动空间分布，用 `magma` colormap 区分（暖色 = 高波动）

**实现**：matplotlib `Figure` → `savefig` PNG。不显示，仅保存文件。每个视频对的可视化独立生成，互不依赖。

**关键设计决策**：
- Heatmap colorbar 范围：固定为 `[0, max(global_threshold * 2, local_threshold * 2)]` 以便跨视频对对比。超出范围的值 clip 到最大值。
- 代表帧选取在 batch_scorer 中完成（已有 per-frame mean ΔE），传给 visualizer。
- PNG 分辨率：源帧原始分辨率，不缩放。Heatmap/mask 同尺寸。

### 模块 2：HTML 报告生成器 (`src/vid_color_filter/report.py`)

纯 Python 字符串模板生成 HTML，无外部依赖。

**单视频对报告页面 (`report.html`)**：
- 顶部：video_pair_id + 数值摘要表格（global_shift_score、local_diff_score、temporal_instability、mask_coverage）
- 帧对比区：左右并排（源帧 vs 编辑帧），可切换帧
- Heatmap 区：ΔE heatmap，可切换帧
- Mask 区：mask overlay，可切换帧
- 时域区：median map 和 IQR map 并排
- 底部：pass/fail 标注按钮

**图片引用**：相对路径（如 `src_frame_0.png`），报告与图片在同一目录。

**索引页 (`index.html`)**：
- 表格：每行一个视频对，列为 video_pair_id、global_shift_score、local_diff_score、temporal_instability、当前标注状态
- 按 global_shift_score 降序排列（最可能 fail 的在前）
- 可按列排序（纯 JS table sort）
- 每行链接到对应 `report.html`
- 顶部显示标注进度（已标注/总数）
- "导出标注" 按钮 → 下载 `annotations.json`

**标注 JS 逻辑**：
- `localStorage` key: `calibration_annotation_{video_pair_id}`
- Value: `{"label": "pass"|"fail", "timestamp": "..."}`
- 导出时遍历所有 `calibration_annotation_*` key，汇总为 JSON 数组
- 页面加载时从 localStorage 恢复已有标注状态
- 标注按钮高亮已选状态（绿色 pass / 红色 fail）

### 模块 3：Calibration 分析 (`src/vid_color_filter/calibration.py`)

**全量分布分析**：
- `global_shift_score` 直方图（50 bins）
- `local_diff_score` 直方图（50 bins）
- 2D 散点图：x=global_shift_score, y=local_diff_score，每个点一个视频对
- `temporal_instability` 直方图
- 输出为 `distribution.html`（matplotlib 图表转 base64 PNG 内嵌）

**网格搜索（标注前预览）**：
- `global_threshold`: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0（19 个值）
- `local_threshold`: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0（15 个值）
- 共 285 组合
- 对每组：统计 pass 数、fail 数、pass rate
- 输出 `grid_search_preview.html`：heatmap 表格，颜色表示 pass rate

**边界案例筛选**：
- 对于搜索范围的中间区域（如 global 1.5~3.0, local 2.0~5.0），找评分落在这些阈值附近（±0.5 ΔE）的视频对
- 去重后作为需要标注的边界案例集合
- 输出 `boundary_cases.json`：包含 video_pair_id 列表和对应评分

**网格搜索（标注后评估）**：
- 输入：`annotations.json`（人工标注的 pass/fail）
- 对每组阈值：以管线判定为预测值，人工标注为真值，计算：
  - Precision：管线判 pass 中真正 pass 的比例
  - Recall：真正 pass 中管线判 pass 的比例
  - F1：precision 和 recall 的调和平均
- 输出 `grid_search_results.html`：F1 heatmap + 最优阈值组合高亮

### 模块 4：CLI 集成 (`run.py`)

新增两个子命令风格的 flag：

```bash
# 步骤 1：跑管线（已有功能），加 --visualize 生成可视化
python run.py --csv data.csv --output calibration_output/scores.jsonl \
    --use-scielab --visualize --viz-dir calibration_output/reports

# 步骤 2：生成分布图 + 网格搜索预览 + 边界案例 + 索引页
python -m vid_color_filter.calibration analyze \
    --scores calibration_output/scores.jsonl \
    --output-dir calibration_output

# 步骤 3（标注后）：加载标注，计算 F1
python -m vid_color_filter.calibration evaluate \
    --scores calibration_output/scores.jsonl \
    --annotations calibration_output/annotations.json \
    --output-dir calibration_output
```

**`--visualize` flag**：
- 在 `batch_scorer.py` 的 `_score_scielab` 中，保留中间产物（代表帧、ΔE map、mask、median/iqr map）
- 调用 `visualizer.py` 生成 PNG
- 调用 `report.py` 生成 HTML
- 仅为边界案例生成可视化（由评分范围决定），避免为全量上千对视频都生成

**性能考虑**：
- 可视化生成是 I/O bound（savefig），不影响 GPU 管线性能
- 每个视频对的可视化互相独立，可并行（但 matplotlib 不是线程安全的，用进程池或串行）
- 生成一个视频对的 5 帧可视化约需 2-3 秒（matplotlib savefig）

### batch_scorer.py 改动

`_score_scielab` 需要在 `--visualize` 模式下额外返回中间产物：

- `representative_frame_indices`: List[int]（代表帧索引）
- `src_frames_repr`: (K, H, W, 3) uint8（代表帧的源 RGB）
- `edit_frames_repr`: (K, H, W, 3) uint8（代表帧的编辑 RGB）
- `de_maps_repr`: (K, H, W) float（代表帧的 ΔE map）
- `masks_repr`: (K, H, W) bool（代表帧的 mask）
- `median_map`: (H, W) float（已有，时域中值 map）
- `iqr_map`: (H, W) float（已有，时域 IQR map）

非 `--visualize` 模式不保留这些中间产物，行为不变。

### 边界案例筛选策略

不在管线运行时判断哪些是边界案例（因为此时还不知道全量分布）。流程为：

1. 管线跑全量，只输出 JSONL（`--visualize` 暂不开）
2. `calibration analyze` 读 JSONL，分析分布，确定边界案例列表
3. **二次运行管线**，仅对边界案例生成可视化：
   ```bash
   python run.py --csv boundary_cases_subset.csv --output /dev/null \
       --use-scielab --visualize --viz-dir calibration_output/reports
   ```
   或者 `calibration analyze` 输出一个子集 CSV，再用 `run.py --csv` 跑

这样避免为全量上千对视频都生成可视化。

## 文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/vid_color_filter/gpu/visualizer.py` | 新建 | 生成 PNG 可视化 |
| `src/vid_color_filter/report.py` | 新建 | 生成 HTML 报告（单页 + 索引 + 标注 JS） |
| `src/vid_color_filter/calibration.py` | 新建 | 分布分析、网格搜索、F1 评估 |
| `src/vid_color_filter/gpu/batch_scorer.py` | 修改 | `--visualize` 模式下保留中间产物 |
| `run.py` | 修改 | 新增 `--visualize` / `--viz-dir` 参数 |
