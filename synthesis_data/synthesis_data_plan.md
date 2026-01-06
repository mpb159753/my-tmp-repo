# 技术设计文档：桌面卡牌堆叠数据合成器 (v3.2 - Realism Update)

## 1. 项目概述
本模块旨在生成“Tacta/Connecta”类桌面卡牌游戏的合成图像数据，用于训练目标检测模型（YOLO）。
* **核心机制**：模拟真实的“堆叠（Stacking）”玩法，即新卡牌（Child）的连接点（Anchor）直接覆盖在旧卡牌（Parent）的目标连接点上。
* **输入前提**：假设已存在经过扩充的单张卡牌数据（包含图像和 JSON 标注）。

## 2. 核心定义：标签与兼容性 (Canonical Schema)

为了消除原始 JSON 标签命名不一致带来的歧义，必须建立**规范化映射（Canonical Mapping）**。系统仅在内部逻辑中使用规范化类型（Canonical Type）。

### 2.1 标签映射表 (Tag Mapping)
在预处理阶段，将 JSON 中的 `label` 映射为以下 4 种核心类型：

| 原始 Label (JSON) | 规范化类型 (Canonical Type) | 几何对称性 (Symmetry) | 备注 |
| :--- | :--- | :--- | :--- |
| `score_1_triangle`<br>`anchor_triangle` | **TYPE_TRIANGLE** | **1-fold** (360°)<br>无对称性，需严格定向 | 位于卡牌角落或边上，局部角度通常为 45°, 135°... |
| `score_2_square`<br>`anchor_square` | **TYPE_SQUARE** | **4-fold** (90°)<br>0°, 90°, 180°, 270° 等价 | 任意 90 度倍数旋转均可重合 |
| `score_4_rect_long`<br>`anchor_rect_long` | **TYPE_RECT_LONG** | **2-fold** (180°)<br>0°, 180° 等价 | 长轴对齐即可 |
| `score_3_rect_short`<br>`anchor_rect_short` | **TYPE_RECT_SHORT** | **2-fold** (180°)<br>0°, 180° 等价 | 长轴对齐即可 |

### 2.2 旋转约束定义
基于卡牌坐标系（X轴向右，Y轴向下），各形状的**局部旋转角 (Local Theta)** 定义如下：
* **Triangle**: 三角形方向计算算法（寻找最大角顶点指向）。**吸附规则**：强制吸附至 45° 的倍数 (0°, 45°, 90°, ...)。
* **Rect/Square**: 沿长轴或特定法线的角度。**吸附规则**：强制吸附至 90° 的倍数 (0°, 90°, 180°, 270°)。

**放置核心逻辑**：
$$\theta_{child\_global} \approx \theta_{target\_global}$$
*(注：对于具有对称性的形状，允许 $\pm 90^\circ$ 或 $\pm 180^\circ$ 的偏差)*

## 3. 增强真实感特征 (Implemented V2 Realism)

### 3.1 变量化光照与阴影 (Variable Lighting Tiers)
为了模拟真实环境的多样性，系统随机分配光照强度等级：
*   **Minimal (30%)**: 极简模式。轻微阴影，无 Vignette，数据干净。
*   **Weak (30%)**: 弱光模式。柔和阴影，轻度暗角。
*   **Medium (25%)**: 中等模式。明显投影，可见光照梯度。
*   **Strong (15%)**: 强光模式。深色长投影，强 Vignette，聚光灯效果，高噪点。

### 3.2 变量化透视畸变 (Variable Perspective)
模拟不同相机拍摄角度：
*   **None (20%)**: 无畸变，正顶视图 (Top-down)。
*   **Normal (50%)**: 轻度倾斜 (10-15% corner shift)。
*   **Heavy (30%)**: 大角度倾斜 (20-35% corner shift)，模拟侧拍。
    *   *关键逻辑*：所有 YOLO 标注均经过透视变换矩阵 (Homography) 更新，并裁剪至可视板面区域内。

### 3.3 真实纹理背景
*   **Textures**: 随机使用高清纹理（深色/浅色木纹、混凝土、绿色毛毡）。
*   **概率**: 80% 概率使用纹理背景，20% 概率使用纯色噪点背景。

## 4. 任务流程 (Pipeline)

### 步骤 1：加载与索引 (Loader)
1.  遍历输入目录中的所有 JSON 文件。
2.  解析每个 Shape，构建 `CardLibrary` 和 `CANDIDATE_INDEX`。

### 步骤 2：合成循环 (Synthesis Engine)
**循环逻辑 (直到卡牌数 >= 5-8)**：
1.  **Pick Target**: 从 `ActiveAnchors` 中选择一个 $A_{target}$。
2.  **Pick Candidate**: 随机获取 $A_{candidate}$。
3.  **Calc Pose**: 计算位姿，应用对称性优化。
4.  **Validate (碰撞检测 - Stricter Rules)**：
    *   **Parent Constraint**: 新卡牌覆盖父卡牌面积 **不得超过 30%**。
    *   **Non-Parent Constraint**: 新卡牌覆盖任何非父卡牌面积 **不得超过 0.5%**（严格防重叠）。
    *   **Anchor Preservation**: 新卡牌 **严禁遮挡** 父卡牌上的任何非目标 Anchor（保留后续连接能力）。
5.  **Commit**: 更新坐标，加入 `PlacedCards`。

### 步骤 3：渲染与后处理 (Renderer)
1.  **Base Render**: 绘制卡牌与阴影（根据 Lighting Tier）。
2.  **Augment**: 应用光照、暗角、噪点（根据 Lighting Tier）。
3.  **Perspective**: 应用透视变换，同步更新几何信息（Polygon & Anchors）。

### 步骤 4：标注生成 (Annotation Generator)
1.  **可见性计算**：
    *   基于透视变换后的几何体。
    *   **Strict Occlusion Check**: 计算未被上层卡牌遮挡的面积。
    *   **Canvas Clipping**: 裁剪至透视变换后的有效板面区域（Valid Poly）。
2.  **过滤规则**：
    *   **Visibility Threshold**: 仅保留可见比例 > **0.5** (50%) 的形状。
    *   **Classes**: 仅输出 Score 1-4。

### 步骤 5：输出
1.  **Images**: `synth_xxxx.jpg`
2.  **Labels**: `synth_xxxx.txt` (YOLO Segmentation format)
    *   Format: `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>` (Normalized 0-1)
    *   Class Map: `1->0, 2->1, 3->2, 4->3`.
    *   **Polygon**: Based on the visible clipping of the anchor shape.

## 5. 配置参数 (Config)
```python
CONFIG = {
    "OUTPUT_DIR": "./synthesis_data/dataset",
    "TOTAL_IMAGES": 5000,
    "MIN_CARDS": 5,
    "MAX_CARDS": 10,
    "COLLISION": {
        "PARENT_OVERLAP_MAX": 0.30,
        "NON_PARENT_OVERLAP_MAX": 0.005,
        "PRESERVE_PARENT_ANCHORS": True
    },
    "VISIBILITY_THRESHOLD": 0.5, # 严格阈值
    
    # 概率分布配置 (Probabilities)
    "LIGHTING_DIST": {
        0: 0.30, # Minimal/Clean
        1: 0.30, # Weak
        2: 0.25, # Medium
        3: 0.15  # Strong
    },
    "PERSPECTIVE_DIST": {
        "NONE": 0.20,
        "NORMAL": 0.50, # 10-15%
        "HEAVY": 0.30   # 20-35%
    },
    
    # 调试选项
    "DEBUG": {
        "SAVE_IMAGES": True, # 输出 debug 图片
        "SAVE_LOGS": True    # 输出生成日志
    },
    
    "CANDIDATE_PATH": None, # 默认使用 global loading，可指定特定目录
    
    # YOLO 类别映射
    "CLASS_MAP": {
        1: 0,  # Triangle
        2: 1,  # Square
        3: 2,  # Rect Short
        4: 3   # Rect Long
    }
}
```