# 时域感知 S-CIELAB 视频色差分析

## 问题陈述

当前的视频色差分析方法，本质上是将传统的图像级色差分析逐帧独立应用：

1. 从每对视频中均匀采样 N 帧
2. 对每帧的未编辑区域计算逐像素 CIEDE2000 ΔE
3. 取每帧的 mean ΔE，再取所有帧中的 max 作为视频得分
4. 基于单一阈值判定 pass/fail

这种方法在视频场景下有严重缺陷：

- **Mean 聚合丢失空间分布信息**：大面积低 ΔE 会掩盖小区域的高 ΔE；反过来，编解码引入的均匀微小色移会拉高 mean
- **Max-of-means 对异常帧过度敏感**：单一异常帧（暗场、场景切换、高压缩区域）就能导致整个视频 fail
- **完全没有利用时域信号**：真正的色差（如调色偏移）在时间上是稳定的，编解码噪声则是帧间随机波动的。这个区分信号完全被忽略了
- **固定的 Mask 阈值**：编辑区域检测使用硬编码的 Lab 距离阈值（5.0），可能会错误地将色差区域标记为编辑区域并排除掉

实验结果证实，许多没有明显色差的视频对在当前方法下未能通过。

## 设计

### Pipeline 总览

```
帧采样（可配置帧数）
  → RGB → Lab 转换
  → 编辑区域 Mask（自适应阈值）
  → S-CIELAB 空间滤波（CSF 卷积，模拟人眼空间感知）
  → CIEDE2000 ΔE（在滤波后的 Lab 图像上计算）
  → 时域聚合（逐像素中值/IQR，分离真色差与编解码噪声）
  → 空间分布分析（全局色移 vs 局部色差）
  → 多维评分 + Pass/Fail 判定
```

### 模块 1：S-CIELAB 空间滤波

**目的**：在计算 ΔE 之前，过滤掉人眼视觉系统无法感知的、空间上孤立的像素级色差。

**算法**（遵循 Zhang & Wandell, 1997）：

1. 将 RGB 转换为 XYZ（需要在 `color_space.py` 中新增 `rgb_to_xyz` 函数）
2. 将 XYZ 转换为 Poirson-Wandell 对立色通道（O1, O2, O3），使用变换矩阵：
   ```
   O1 =  0.9795 X + 1.5318 Y + 0.1225 Z   （无彩色/亮度）
   O2 = -0.1071 X + 0.3122 Y + 0.0215 Z   （红-绿）
   O3 =  0.0383 X + 0.0023 Y + 0.5765 Z   （蓝-黄）
   ```
3. 对每个对立色通道独立应用对比敏感度函数（CSF）卷积核。每个 CSF 是 2-3 个高斯函数的加权和（空间域）：
   - **O1（无彩色）**：`CSF(r) = 0.921 * G(r, σ₁) + 0.105 * G(r, σ₂) - 0.026 * G(r, σ₃)`，其中 σ₁=0.0283°, σ₂=0.133°, σ₃=4.336°（视角度数，通过 `pixels_per_degree` 转换为像素）
   - **O2（红-绿）**：`CSF(r) = 0.531 * G(r, σ₁) + 0.330 * G(r, σ₂)`，其中 σ₁=0.0392°, σ₂=0.494°
   - **O3（蓝-黄）**：`CSF(r) = 0.488 * G(r, σ₁) + 0.371 * G(r, σ₂)`，其中 σ₁=0.0536°, σ₂=0.386°
   - 核大小：在最大高斯分量的 3σ 处截断，必须为奇数
   - 所有核归一化为总和 1.0
4. 将滤波后的对立色通道转换回 XYZ（上述矩阵的逆变换）
5. 将 XYZ 转换为 Lab
6. 在滤波后的 Lab 图像上计算 CIEDE2000 ΔE

**关键参数**：`--pixels-per-degree`（默认：60，对应桌面显示器 ~60cm 观看距离）。视角度数的 σ 值乘以此参数即转换为像素 σ 值。

**GPU 实现**：对每个分辨率/观看距离组合预计算一次 CSF 核，缓存为 tensor。通过 `F.conv2d` 并设 padding='same' 进行三通道 2D 卷积，开销很小。

**效果**：编解码引入的随机逐像素色彩噪声被平滑掉。只有空间上连贯的色差（人眼可感知的）保留在 ΔE map 中。

### 模块 2：自适应编辑区域 Mask

**当前方法**：固定 Lab 距离阈值 5.0 + 形态学膨胀。

**问题**：固定阈值可能将合理的色差区域错误标记为"编辑区域"而排除，或者漏掉细微的编辑。

**新方法**：

1. 计算源帧和编辑帧之间的逐像素 Lab 欧几里得距离（与当前相同）
2. **Otsu 自适应阈值**：从距离直方图自动确定二值化阈值，适应每帧的内容
3. **滞后阈值**（双阈值）：
   - 高阈值（来自 Otsu）：种子区域——确定是编辑区域
   - 低阈值（Otsu × 0.5）：扩展——与种子区域相连且高于此阈值的区域也标记为编辑
   - 类似 Canny 边缘检测的逻辑
4. 使用现有膨胀核进行形态学膨胀作为安全边界

**GPU 实现**：`torch.histc` 计算直方图，Otsu 阈值通过类间方差最大化实现。滞后扩展通过在种子 mask 上迭代 `F.max_pool2d` 并结合低阈值门控实现——迭代直到收敛（没有新像素加入），最多 50 次迭代作为计算上限。

### 模块 3：时域聚合

**目的**：区分时间上稳定的色差（真实色差）和帧间波动（编解码噪声）。

对每个像素位置 (x, y)，跨所有 N 个采样帧计算：

- **像素 Mask 策略**：一个像素只有在至少 50% 的采样帧中未被 mask 时才纳入时域聚合。时域中值/IQR 仅在该像素未被 mask 的帧子集上计算。被 mask 超过 50% 帧的像素从评分中排除。
- **时域中值 ΔE(x, y)**：该位置的稳定色差信号。对编解码帧间波动具有鲁棒性。
- **时域 IQR(x, y)**：跨帧 ΔE 的四分位距。衡量色差的波动程度。

判读规则：

| 中值 | IQR | 解读 |
|------|-----|------|
| 高 | 低 | 真色差（跨帧稳定存在） |
| 低 | 高 | 编解码噪声（随机波动） |
| 高 | 高 | 不确定——标记待人工审查 |
| 低 | 低 | 无色差 |

**输出**：两个空间 map——`median_map(H, W)` 和 `iqr_map(H, W)`——分别表示稳定色差信号及其可靠性。

**GPU 实现**：沿帧维度使用 `torch.median` 和 `torch.quantile`。需要所有 N 帧的 ΔE map 同时在显存中（形状：`(N, H, W)`）。对于跨帧 mask 状态不同的像素，将被 mask 的值设为 NaN，使用 `torch.nanmedian`/`torch.nanquantile`。

### 模块 4：空间分布分析与多维评分

从时域中值 ΔE map（仅未 mask 像素）计算三个分数：

**1. 全局色移分数（Global Shift Score）**
- `median(temporal_median_map)` — 所有未 mask 像素的稳定 ΔE 值的中值
- 代表整帧的均匀色偏（如编解码引入的整体偏暖/偏冷）
- 阈值：`--global-threshold`（默认：2.0 ΔE）

**2. 局部色差分数（Local Difference Score）**
- `P95(temporal_median_map) - median(temporal_median_map)` — 最差 5% 与全局水平的偏差
- 代表高于全局基线的局部色差异常
- 阈值：`--local-threshold`（默认：3.0 ΔE）

**3. 时域不稳定性分数（Temporal Instability Score）**
- `mean(temporal_iqr_map)` — 帧间波动的平均值
- 高值说明色差主要来自编解码噪声而非真实色移
- 作为元数据/置信度指标报告，默认不参与 pass/fail 判定

**Pass/Fail 逻辑**：
```
pass_global = global_shift_score < global_threshold
pass_local  = local_diff_score  < local_threshold
pass        = pass_global AND pass_local
```

### 输出格式

```json
{
  "video_pair_id": "clip_001",
  "global_shift_score": 0.8,
  "local_diff_score": 0.3,
  "temporal_instability": 0.15,
  "pass_global": true,
  "pass_local": true,
  "pass": true,
  "mask_coverage_ratio": 0.23,
  "per_frame_mean_delta_e": [0.12, 0.15, 0.11, ...]
}
```

- `per_frame_mean_delta_e` 保留以向后兼容（基于 S-CIELAB 滤波后的 ΔE 计算）
- `mask_coverage_ratio` 保留（取所有帧的最大值）

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num-frames` | 32 | 每视频采样帧数。从 16 增加到 32 以支持时域统计。 |
| `--pixels-per-degree` | 60 | S-CIELAB 的观看条件参数。对应桌面显示器 ~60cm。 |
| `--global-threshold` | 2.0 | 全局色移分数的 ΔE 阈值 |
| `--local-threshold` | 3.0 | 局部色差分数的 ΔE 阈值 |
| `--threshold` | — | `--global-threshold` 的别名（向后兼容） |
| `--metric` | `ciede2000` | 默认从 `cie94` 改为 `ciede2000`。CIE76/CIE94 仍可用。 |
| `--diff-threshold` | `None` | 编辑 Mask 阈值。`None` = Otsu 自适应（新默认值）。传入浮点数则使用固定阈值（旧行为）。 |
| `--dilate-kernel` | 21 | Mask 膨胀核大小（不变） |
| `--chunk-size` | 8 | 每个 GPU 处理块的帧数。降低此值可减少高分辨率视频的显存占用。 |

### GPU 兼容性

所有新增计算都有原生 PyTorch 实现：

| 操作 | 实现方式 |
|---|---|
| S-CIELAB CSF 卷积 | 预计算核 + `F.conv2d` |
| Otsu 阈值 | `torch.histc` + 类间方差 argmax |
| 滞后扩展 | 迭代 `F.max_pool2d` + mask 门控 |
| 时域中值 | `torch.median`（dim=0） |
| 时域 IQR | `torch.quantile`（Q75 - Q25，dim=0） |
| 空间百分位数 | `torch.quantile`（展平后） |

**显存预算**（1080p，32 帧，float32）：

| Tensor | 形状 | 大小 |
|---|---|---|
| 源 + 编辑后 Lab 帧 | 2 × 32 × 1080 × 1920 × 3 | ~1.5 GB |
| S-CIELAB 滤波后 Lab（双份） | 2 × 32 × 1080 × 1920 × 3 | ~1.5 GB |
| 逐像素 ΔE map | 32 × 1080 × 1920 | ~250 MB |
| Mask（膨胀时为 float） | 32 × 1080 × 1920 | ~250 MB |
| **峰值总计** | | **~3.5 GB** |

对于 4K（2160×3840），峰值显存约为 4 倍即 ~14 GB。为支持显存有限情况下的大分辨率处理，pipeline 支持**分块处理**：以 `--chunk-size`（默认：8）帧为单位分块处理，增量累积逐像素 ΔE map。这是用吞吐量换显存。无需分块的最大支持分辨率：8 GB GPU 上 1080p，24 GB GPU 上 4K。

### 向后兼容性

**破坏性变更**（已记录）：
- `--num-frames` 默认值从 16 → 32（解码时间和显存翻倍）。用户可传 `--num-frames 16` 恢复旧行为。
- `--metric` 默认值从 `cie94` → `ciede2000`（更准确但更慢）。用户可传 `--metric cie94` 恢复旧行为。
- `--diff-threshold` 默认值从 `5.0` → `None`（Otsu 自适应）。用户可传 `--diff-threshold 5.0` 恢复旧行为。
- 输出字段 `max_mean_delta_e` 被 `global_shift_score` 替代。旧字段名保留一个版本周期作为别名。

**保留不变**：
- CIE76/CIE94 metric 选项通过 `--metric` 仍可使用
- `--threshold` 仍作为 `--global-threshold` 的别名
- 输出中同时保留 `per_frame_mean_delta_e` 和 `mean_delta_e_per_frame`（相同数据，两个字段名）
- `mask_coverage_ratio` 保留
- CPU pipeline（`cli.py`）不变；新功能仅在 GPU pipeline 中

### 需要修改/创建的文件

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/vid_color_filter/gpu/color_space.py` | 修改 | 新增 `rgb_to_xyz` 函数，S-CIELAB 对立色转换需要 |
| `src/vid_color_filter/gpu/color_metrics.py` | 修改 | 新增逐像素 ΔE 模式（返回 `(B, H, W)` map 而非 `(B,)` 均值），用于时域聚合 |
| `src/vid_color_filter/gpu/scielab.py` | 创建 | S-CIELAB 空间滤波（CSF 核、Poirson-Wandell 对立色转换） |
| `src/vid_color_filter/gpu/adaptive_mask.py` | 创建 | Otsu + 滞后阈值自适应 Mask 生成 |
| `src/vid_color_filter/gpu/temporal_aggregator.py` | 创建 | 时域中值/IQR 聚合与多维评分 |
| `src/vid_color_filter/gpu/batch_scorer.py` | 修改 | 集成新模块到评分 pipeline |
| `run.py` | 修改 | 新增 CLI 参数 |
| `tests/test_scielab.py` | 创建 | S-CIELAB 正确性测试 |
| `tests/test_adaptive_mask.py` | 创建 | 自适应 Mask 测试 |
| `tests/test_temporal_aggregator.py` | 创建 | 时域聚合测试 |
| `tests/test_gpu_scorer.py` | 修改 | 更新端到端测试以适配新输出格式 |
