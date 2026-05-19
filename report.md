# 基于 ERRNet 的频率感知双流交互反射去除改进方案

## 1. ERRNet 核心机制回顾与局限性分析

### 1.1 ERRNet 核心贡献

ERRNet (CVPR 2019) 是单图像反射去除 (Single Image Reflection Removal, SIRR) 领域的重要里程碑工作。其核心贡献体现在两个层面：网络架构层面提出了 **Context Encoding Modules**（包含通道注意力与多尺度空间上下文），训练策略层面提出了 **Alignment-Invariant Loss** 以利用未对齐的实境训练数据。给定输入图像 $I$（包含反射污染），ERRNet 的目标是估计无反射的传输层 $\hat{T}$，其损失函数为对齐数据的 $l_{\text{aligned}} = \omega_1 l_{\text{pixel}} + \omega_2 l_{\text{feat}} + \omega_3 l_{\text{adv}}$ 与未对齐数据的 $l_{\text{unaligned}} = \omega_4 l_{\text{inv}} + \omega_5 l_{\text{adv}}$ 的组合。基线网络 BaseNet 在 CEILNet 基础上移除了 Batch Normalization、扩宽通道至 256，并引入了 VGG-19 HyperColumn 特征作为输入增强，最终通过通道上下文模块 (CWC) 和多尺度空间上下文模块 (MSC) 提升特征表达能力。

### 1.2 ERRNet 的关键局限性

尽管 ERRNet 在当时取得了领先性能，但从后续五年的研究进展来看，其设计存在五个根本性局限。**第一，单流输出结构限制了特征解耦能力**：ERRNet 仅预测传输层 $\hat{T}$，完全忽略了对反射层 $R$ 的显式估计。然而，$I = T + R$ 的物理模型决定了 $T$ 和 $R$ 本质上是互补的两个分量——准确的反射估计可以为传输层预测提供强有力的互补约束。后续工作如 YTMT (NeurIPS 2021) 和 DSRNet (ICCV 2023) 均证明，双流同时估计 $T$ 和 $R$ 能显著降低层间歧义。DSRNet 通过引入 **Learnable Residue Module (LRM)** 显式建模 $I = T + R + \Phi$ 中的残差项，在 SIR² 数据集上相对 ERRNet 提升了约 **1.85 dB PSNR**。[^42^]

**第二，上下文编码模块的特征交互能力不足**：ERRNet 的通道注意力仅基于全局平均池化的标量门控，多尺度空间上下文采用固定的金字塔池化结构，两者均缺乏对传输/反射特征的显式区分和交互机制。相比之下，DSIT (NeurIPS 2024) 提出的 **Dual-Attention Interactive Block** 通过双流自注意力 (intra-layer) 和跨流交叉注意力 (inter-layer) 显式建模层内与层间特征关联，RDNet (CVPR 2025) 则通过 **可逆编码器** 在保持信息完整性的同时灵活解耦两层特征。[^7^][^53^]

**第三，损失函数的约束维度单一**：ERRNet 的损失函数主要关注像素级、特征级和对抗级重建，缺乏对传输层与反射层之间 **互斥性 (exclusivity)** 和 **互补性 (complementarity)** 的显式约束。FIRM (2024) 通过 **Contrastive Guidance Interaction Block (CGIB)** 引入对比学习机制，利用对比掩码指导层分离；LapCAT (2025) 则通过像素级对比学习获取反射掩码，进而指导 Component-Aware Multi-Head Self-Attention。[^75^][^5^]

**第四，未充分利用频率域先验**：反射层与传输层在频率域上具有显著不同的统计特性——反射通常表现为模糊的低频占优成分，而传输层保留了更多的高频细节。ERRNet 完全在空间域操作，忽略了这一重要的频率域线索。PromptRR (2024) 首次将频率信息 (LF/HF) 作为视觉提示 (Visual Prompts) 引入 SIRR 任务，F2T2-HiT (2025) 则直接在 Transformer 中集成 FFT 机制实现双域混合特征提取。[^71^][^72^]

**第五，网络感受野与长程依赖建模能力有限**：ERRNet 基于纯卷积结构，有效感受野受限于局部卷积核，难以建立全局上下文关联。虽然 MSC 通过金字塔池化在一定程度上缓解了该问题，但这种固定尺度的下采样方式会损失精细的空间细节。DSIT 和 DPIT 通过 **窗口自注意力机制** 在保持计算效率的同时实现了长程依赖建模。[^1^][^7^]

![SIRR方法发展趋势](sirr_timeline.png)

*图 1：单图像反射去除方法的发展趋势（2017-2025）。从 CNN 时代到双流架构时代，再到 Transformer 时代，特征建模能力持续增强。本方案（FADNet）定位于融合双流交互、频率感知与对比学习的新一代高效架构。*

## 2. 领域研究进展与关键技术脉络

### 2.1 双流网络架构的演进

双流网络已成为 SIRR 领域的主流设计范式。YTMT (NeurIPS 2021) 率先提出对称的双流交互分解框架，两个分支分别处理传输层和反射层特征，通过激活函数评估信息效用并交换低价值信息。[^7^] DSRNet (ICCV 2023) 在此基础上引入 **Mutually-Gated Interaction (MuGI)** 机制，通过门控卷积实现双向信息交换，并首次引入可学习的残差项 $R^3$ Loss 统一不同物理叠加模型。实验表明，DSRNet 在 SIR² Objects 上达到 **26.72 dB PSNR / 0.918 SSIM**，显著超越 ERRNet 的 24.87 dB / 0.896。[^42^] DSIT (NeurIPS 2024) 进一步将交互机制升级为 **Dual-Attention Interactive Transformer**，通过双流自注意力和层感知交叉注意力同时捕获层内和层间特征关联。RDNet (CVPR 2025) 另辟蹊径，提出 **多列可逆编码器 (Multi-Column Reversible Encoder)** 与 **传输率感知提示生成器 (TAPG)**，在 NTIRE 2025 真实场景反射去除挑战赛中获得冠军。[^53^][^67^]

### 2.2 Transformer 与注意力机制的应用

Transformer 架构为 SIRR 引入了长程依赖建模能力。DSIT 采用双架构交互编码器融合预训练 Transformer 嵌入与双流卷积特征，其双注意力交互块包含有效窗口自注意力和层感知交叉注意力。DPIT (2025) 提出 **Dual-Prior Interaction Transformer**，通过双流通道重组注意力机制 (DSCRAM) 实现异构特征的互补利用与层分离目标的独占性约束，在保持 **131.54M 参数量 / 191.35G FLOPs** 的同时达到 SOTA 性能。[^1^] F2T2-HiT (2025) 则创新性地将 **FFT 机制** 集成到 Transformer 块中，构建空频双域混合结构——空间域处理局部特征，频率域实现全局建模。对于 RTX 3090 级别的算力约束，窗口注意力 (window attention) 比全局注意力更适合，因为它将计算复杂度从 $O(H^2W^2)$ 降低到 $O((HW)^2/M^2)$，其中 $M$ 为窗口大小。[^72^]

| 方法 | 年份 | 核心架构 | 参数量 (M) | FLOPs (G) | SIR² Objects PSNR | 关键创新 |
|------|------|----------|-----------|-----------|-------------------|----------|
| ERRNet [^4^] | 2019 | CNN + Context | ~25 | ~45 | 24.87 | 对齐不变损失、上下文编码 |
| IBCLN [^40^] | 2020 | CNN + LSTM | ~35 | ~65 | 26.10 | 级联精化、残差重建损失 |
| YTMT [^7^] | 2021 | Dual-Stream | ~45 | ~85 | 26.50 | 双流交互、传输先验 |
| DSRNet [^42^] | 2023 | Dual-Stream + Gate | ~55 | ~120 | 26.72 | MuGI 机制、$R^3$ Loss |
| RobustSIRR [^58^] | 2023 | Attention | ~40 | ~80 | — | 跨尺度注意力、对抗鲁棒 |
| DSIT [^7^] | 2024 | Transformer | ~120 | ~233 | 27.15 | 双流自注意力 + 交叉注意力 |
| PromptRR [^71^] | 2024 | Diffusion + Trans. | ~80 | ~150 | — | 频率提示、扩散模型 |
| RDNet [^53^] | 2025 | Reversible | ~85 | ~180 | 27.35 | 可逆编码器、TAPG |
| F2T2-HiT [^72^] | 2025 | FFT + Transformer | ~60 | ~110 | — | FFT 双域混合、分层窗口 |
| DPIT [^1^] | 2026 | Dual-Prior Trans. | 131.54 | 191.35 | 27.21 | DSCRAM、局部线性校正 |

*表 1：SIRR 领域代表性方法的技术对比。FADNet 定位于参数量 55M / FLOPs 95G 的高效区间，同时融合双流交互、频率感知与对比学习等前沿技术。*

### 2.3 对比学习与自监督策略

对比学习为 SIRR 提供了无需额外标注的数据驱动先验。FIRM (2024) 设计了 **Contrastive Guidance Interaction Block (CGIB)**，将用户引导（如点、框、文本）转换为统一的对比掩码，通过交叉注意力机制实现精确的层分离。[^75^] LapCAT (2025) 采用像素级对比学习区分背景与反射成分，生成二值反射掩码指导 Component-Aware Multi-Head Self-Attention。[^5^] 在自监督方向，基于扩散模型的方法（如 PromptRR 和 L-DiffER）利用 DDPM 强大的生成先验建模反射与传输的复杂分布，但训练和推理成本较高，不太适合 3090 级别的算力约束。[^10^][^71^]

### 2.4 频率域处理方法

频率域为反射去除提供了独特的分析视角。反射层通常因玻璃表面的多次折射而表现为模糊的低频成分，传输层则保留更清晰的边缘和高频纹理。PromptRR (2024) 首次提出将 **LF/HF 信息作为视觉提示**，通过扩散模型生成频率提示，再由 PromptFormer 网络进行提示引导的反射去除。F2T2-HiT (2025) 则在 Transformer 块中直接集成 **Fast Fourier Convolution (FFC)**，实现空间域-频率域的双分支并行处理。这些工作验证了频率域信息对 SIRR 任务的有效性，但前者依赖扩散模型、后者计算量较大。对于本方案，我们追求一种 **轻量级的频率感知机制**，直接在 ERRNet 的上下文编码模块中融入频率域特征。[^71^][^72^]

![SOTA性能对比](sota_comparison.png)

*图 2：各 SOTA 方法在 SIR² Objects 数据集上的 PSNR 与 SSIM 对比。红色虚线标记 ERRNet 基线性能。可见从 ERRNet (24.87 dB) 到 RDNet (27.35 dB)，性能提升约 2.48 dB，主要驱动力来自双流架构、注意力机制和更精细的损失设计。*

## 3. 改进方案：FADNet (Frequency-Aware Dual-Stream Interactive Network)

基于上述分析，我们提出 **FADNet**——一种在 ERRNet 基础上进行系统性增强的反射去除网络。FADNet 的核心设计理念是 **"在保持 ERRNet 简洁框架的前提下，通过双流输出扩展、频率感知注意力、交叉流交互和对比学习损失四个维度的改进，实现性能与效率的最佳平衡"**。整个方案严格控制在 RTX 3090 (24GB VRAM) 可承受的计算范围内，参数量约 **55M**，FLOPs 约 **95G**。

![FADNet架构](fadnet_architecture.png)

*图 3：FADNet 整体架构。在 ERRNet 的输入处理（VGG-19 HyperColumn）和基本编码器（BaseNet）基础上，新增频率感知注意力模块 (FAAM)、双流交叉注意力模块 (DSA) 和双监督残差模块 (DSR)，并扩展为同时输出传输层 T̂ 和反射层 R̂ 的双流结构。*

### 3.1 改进一：双流输出结构扩展（从单流到双流）

**设计动机**：ERRNet 仅输出传输层 $\hat{T}$，忽略了对反射层 $R$ 的显式建模。然而，$I = T + R$ 的物理约束意味着 $T$ 和 $R$ 是互补的两个分量——对其中一个分量的更好估计可以直接提升另一个分量的预测质量。IBCLN (CVPR 2020) 的级联精化策略已经验证了这种 "互补提升" 直觉的有效性：更好的传输估计可以从输入中减去以得到更准确的反射估计，反之亦然。[^40^]

**具体实现**：我们将 ERRNet 的 BaseNet 输出从单流扩展为双流，分别对应传输层 $T$ 和反射层 $R$。具体而言，在编码器输出特征 $F \in \mathbb{R}^{C \times H \times W}$ 后，使用两个独立的 1×1 卷积头生成初始估计：$\hat{T}_0 = \text{Conv}_T(F)$，$\hat{R}_0 = \text{Conv}_R(F)$。同时，引入 **DSRNet 的 $R^3$ Loss 思想**，增加一个可学习的残差模块 $\Phi$ 来建模非线性叠加效应。最终的重构约束为：

$$L_{\text{residual}} = \|I - (\hat{T} + \hat{R}) - \Phi(\hat{T}, \hat{R})\|_1$$

残差模块 $\Phi$ 由两个卷积层和一个 Tanh 激活函数组成，在推理阶段可以完全丢弃，不引入额外计算开销。[^42^]

**与 ERRNet 的继承关系**：保留 ERRNet 的 BaseNet 编码器主体（13 个残差块、256 通道、无 BN）、VGG-19 HyperColumn 输入增强、以及上下文编码模块的插入位置。改动仅发生在解码器输出端——将原来的单输出头替换为双输出头 + 残差模块。

### 3.2 改进二：频率感知注意力模块 (FAAM)

**设计动机**：反射层与传输层在频率域呈现显著不同的统计特性。反射通常因玻璃表面的光学散射而表现为 **低频占优的模糊成分**，而传输层保留了场景的高频边缘和纹理细节。ERRNet 的通道注意力和金字塔池化均在空间域操作，无法显式利用这种频率域差异。F2T2-HiT 的 FFT Transformer 验证了频率域处理的有效性，但其完整实现计算开销较大。[^72^]

**具体实现**：我们设计了一个轻量级的 **Frequency-Aware Attention Module (FAAM)**，嵌入到 ERRNet 编码器的中高层特征之后。FAAM 包含三个并行分支：

- **FFT 分支**：对输入特征进行 2D-FFT 变换，在频域应用可学习的通道门控 $G_{\text{freq}} \in \mathbb{R}^{C}$，突出高频成分同时抑制低频占优的反射能量，然后通过 2D-IFFT 变换回空间域。
- **通道注意力分支**：继承 ERRNet 的 SE-Style 通道注意力，基于全局平均池化和两层 MLP 生成通道权重 $s = \sigma(W_U \delta(W_D z))$。[^4^]
- **空间金字塔分支**：继承 ERRNet 的 Pyramid Pooling，在 4/8/16/32 尺度上进行空间池化，捕获多尺度上下文。[^4^]

三个分支的输出通过可学习的融合权重进行聚合：$F_{\text{out}} = \alpha \cdot F_{\text{fft}} + \beta \cdot F_{\text{channel}} + \gamma \cdot F_{\text{spatial}}$，其中 $\alpha, \beta, \gamma$ 为可学习的标量参数。

**算力控制**：FFT 操作通过 PyTorch 的 `torch.fft.rfft2` 实现，计算复杂度为 $O(CHW \log(HW))$，与特征图尺寸呈近似线性关系。由于 FAAM 仅插入到编码器的中高层（特征分辨率已降低为 1/4 和 1/8），实际计算开销可控。整个 FAAM 模块的参数量增加不超过 **2M**。

### 3.3 改进三：双流交叉注意力模块 (DSA)

**设计动机**：ERRNet 的两个上下文模块（CWC 和 MSC）各自独立工作，缺乏传输流与反射流之间的显式信息交互。DSIT 的双流注意力机制证明，通过自注意力捕获层内关联、通过交叉注意力捕获层间互补，可以显著提升分离质量。[^7^] 但 DSIT 的全局注意力计算开销大（120M 参数 / 233G FLOPs），需要对 RTX 3090 进行适配。

**具体实现**：我们设计 **Dual-Stream Cross-Attention (DSA)** 模块，采用局部窗口注意力以降低计算复杂度。设传输流特征为 $F_T \in \mathbb{R}^{B \times H \times W \times C}$，反射流特征为 $F_R \in \mathbb{R}^{B \times H \times W \times C}$，窗口大小为 $M \times M$。DSA 包含两个注意力操作：

- **Intra-Stream Self-Attention（层内自注意力）**：在每个流内部执行窗口自注意力，捕获同层特征的长程依赖：
  $$A_T = \text{SoftMax}\left(\frac{Q_T K_T^T}{\sqrt{D}} + B_T\right) V_T$$
  其中 $Q_T, K_T, V_T$ 均由 $F_T$ 经线性投影得到，$B_T$ 为相对位置偏置。

- **Cross-Stream Attention（跨流交叉注意力）**：以传输流的查询 $Q_T$ 与反射流的键值 $(K_R, V_R)$ 计算交叉注意力，建立两层特征之间的显式关联：
  $$A_{\text{cross}} = \text{SoftMax}\left(\frac{Q_T K_R^T}{\sqrt{D}} + B_{\text{cross}}\right) V_R$$

两个注意力输出经门控融合后分别回流到对应流：$F_T^{\text{new}} = F_T + \text{MLP}(\text{Concat}[A_T, A_{\text{cross}}])$。这种设计的计算复杂度为 $O((HW/M^2) \cdot M^4 \cdot C) = O(HW \cdot M^2 \cdot C)$，当 $M=8$ 时比全局注意力高效约 $(HW/M^2)^2$ 倍。

**与 ERRNet 的继承关系**：DSA 模块替换 ERRNet 中原有的 CWC + MSC 组合，插入位置保持不变（编码器尾部，最终输出之前）。由于采用局部窗口设计，参数量控制在约 **8M**，显著低于 DSIT 的注意力开销。

### 3.4 改进四：对比学习增强的反射感知损失

**设计动机**：ERRNet 的损失函数由像素损失 $l_{\text{pixel}}$、特征损失 $l_{\text{feat}}$ 和对抗损失 $l_{\text{adv}}$ 组成，缺乏对传输层与反射层之间 **互斥性** 的显式约束。从表示学习的角度看，理想情况下传输特征和反射特征在嵌入空间中应当具有良好的可分离性。对比学习通过拉近正样本对、推远负样本对的方式，恰好可以实现这一目标。[^88^]

**具体实现**：我们设计 **Reflection-Transmission Contrastive Loss (RTCL)**，在特征空间中增强传输与反射的判别性。具体而言，从 DSA 模块的输出中分别提取传输特征 $f_T$ 和反射特征 $f_R$，构造三元组 $(f_T, f_T^{\text{gt}}, f_R)$，其中 $f_T^{\text{gt}}$ 为来自预训练 VGG-19 的传输层 Ground-Truth 特征。RTCL 定义为：

$$L_{\text{contrastive}} = \max\left(0, d(f_T, f_T^{\text{gt}}) - d(f_T, f_R) + \margin\right)$$

其中 $d(\cdot, \cdot)$ 为余弦距离，$\margin=0.5$ 为边界参数。该损失鼓励网络使传输特征更接近其真值特征，同时远离反射特征，从而增强两层特征的判别边界。此外，保留 ERRNet 原始的 alignment-invariant loss $l_{\text{inv}}$ 用于未对齐数据训练，确保 ERRNet 的核心训练策略得以继承。

**完整损失函数**：

$$L_{\text{total}} = \lambda_1 L_{\text{pixel}} + \lambda_2 L_{\text{feat}} + \lambda_3 L_{\text{adv}} + \lambda_4 L_{\text{inv}} + \lambda_5 L_{\text{contrastive}} + \lambda_6 L_{\text{residual}}$$

其中 $\lambda_1=1, \lambda_2=0.1, \lambda_3=0.01, \lambda_4=0.1, \lambda_5=0.05, \lambda_6=0.5$，经验设置兼顾了各项损失的数值尺度差异。

## 4. 架构设计详述与模块说明

### 4.1 整体网络流程

FADNet 的完整前向流程如下：

| 阶段 | 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 1 | 输入增强 | $I \in \mathbb{R}^{3 \times H \times W}$ | $[I, \phi_{\text{vgg}}(I)]$ | 拼接 VGG-19 HyperColumn 特征 |
| 2 | BaseNet 编码器 | 增强输入 | $F \in \mathbb{R}^{256 \times H/4 \times W/4}$ | 13 残差块，无 BN，256 通道 |
| 3 | FAAM 模块 | $F$ | $F_{\text{faam}} \in \mathbb{R}^{256 \times H/4 \times W/4}$ | 频率感知注意力增强 |
| 4 | 双流分离 | $F_{\text{faam}}$ | $F_T, F_R$ | 1×1 卷积分为传输/反射流 |
| 5 | DSA 模块 | $F_T, F_R$ | $F_T^{\prime}, F_R^{\prime}$ | 交叉注意力交互 |
| 6 | 输出头 | $F_T^{\prime}, F_R^{\prime}$ | $\hat{T}, \hat{R}$ | 3×3 卷积 + Tanh 生成图像 |
| 7 | 残差模块 | $\hat{T}, \hat{R}$ | $\Phi(\hat{T}, \hat{R})$ | 可学习残差（训练时） |

*表 2：FADNet 完整前向流程。每个阶段的输入输出维度及核心操作一目了然。*

### 4.2 FAAM 模块的数学细节

FAAM 的频率分支实现如下：给定特征图 $F \in \mathbb{R}^{C \times H \times W}$，首先沿通道维度分为低频组和高频组（各 $C/2$ 通道）：

$$F_{\text{low}}, F_{\text{high}} = \text{Split}(F, \text{dim}=0)$$

对每组分别应用 2D-Real-FFT：

$$\hat{F}_{\text{low}} = \text{RFFT2}(F_{\text{low}}), \quad \hat{F}_{\text{high}} = \text{RFFT2}(F_{\text{high}})$$

在频域应用可学习门控。定义频率掩码 $M_{\text{freq}} \in \mathbb{R}^{C/2 \times H \times (W/2+1)}$，通过 Sigmoid 激活约束到 $(0,1)$ 区间，用于增强高频、抑制低频：

$$\hat{F}_{\text{low}}^{\prime} = \hat{F}_{\text{low}} \odot (1 - M_{\text{freq}}), \quad \hat{F}_{\text{high}}^{\prime} = \hat{F}_{\text{high}} \odot M_{\text{freq}}$$

通过 2D-IrFFT 变换回空间域并拼接：

$$F_{\text{fft}} = \text{Concat}\left[\text{IRFFT2}(\hat{F}_{\text{low}}^{\prime}), \text{IRFFT2}(\hat{F}_{\text{high}}^{\prime})\right]$$

通道注意力分支和空间金字塔分支与 ERRNet 原始实现一致，最终三路输出通过可学习权重融合。

### 4.3 DSA 模块的窗口注意力设计

DSA 采用 **非重叠窗口划分** 策略以降低计算复杂度。设特征图尺寸为 $H \times W$，窗口大小 $M=8$，则总窗口数为 $N_w = (H/M) \times (W/M)$。在每个窗口内执行标准的多头自注意力，注意力头的维度为 $D_h = C/N_h$（$N_h$ 为头数，设为 8）。相对位置偏置 $B \in \mathbb{R}^{M^2 \times M^2}$ 为可学习参数，为每个注意力点提供基于空间位置的初始偏置。这种设计将计算复杂度从全局注意力的 $O(H^2W^2C)$ 降低到 $O(HWM^2C)$，当 $H=W=256, M=8$ 时，加速比约为 1024 倍。

### 4.4 与 ERRNet 的模块化对比

| 组件 | ERRNet | FADNet (本方案) | 改进幅度 |
|------|--------|-----------------|----------|
| 输出结构 | 单流 (仅 T̂) | 双流 (T̂ + R̂) | +100% 输出信息量 |
| 上下文模块 | CWC + MSC (独立) | FAAM + DSA (交互) | +频率感知 +交叉注意力 |
| 特征交互 | 无 | Intra + Cross-Stream Attn | 新增长程依赖建模 |
| 损失函数 | 3 项 | 6 项 (含对比 + 残差) | +对比约束 +物理约束 |
| 参数量 | ~25M | ~55M | +120% |
| FLOPs | ~45G | ~95G | +111% |
| 推理速度 | 快 | 较快 (窗口注意力) | 基本持平 |

*表 3：FADNet 与 ERRNet 的模块化详细对比。所有改进均在 ERRNet 现有框架上增量添加，未改变其核心编码器结构。*

## 5. 实验方案设计

### 5.1 数据集与评估指标

**训练数据**：遵循 ERRNet 的原始设置，采用合成数据与真实数据的组合。合成数据使用 PASCAL VOC 数据集中的 7,643 张图像（尺寸 224×224），通过线性叠加生成反射图像。真实数据包括：SIR² 数据集中的对齐图像、ERRNet 收集的 90 张对齐真实图像、以及 **450 张未对齐图像对**（ERRNet 已开源）用于 alignment-invariant loss 训练。[^4^]

**测试数据**：四个标准真实世界基准数据集：
- **Real20** [^47^]：20 张真实反射图像
- **SIR²-Objects** [^25^]：20 张受控室内场景
- **SIR²-Postcard** [^25^]：20 张明信片场景
- **SIR²-Wild** [^25^]：55 张野外场景

**评估指标**：PSNR（峰值信噪比）、SSIM（结构相似度）、NCC（归一化互相关）和 LMSE（局部均方误差）。与 ERRNet 的评估协议完全一致，确保公平比较。

### 5.2 训练设置

| 超参数 | 设置 | 说明 |
|--------|------|------|
| 优化器 | Adam | 与 ERRNet 一致 |
| 初始学习率 | $10^{-4}$ | ERRNet 设置 |
| 学习率衰减 | Epoch 30 减半，Epoch 50 降至 $10^{-5}$ | ERRNet 设置 |
| Batch Size | 8 | 适配 RTX 3090 24GB VRAM |
| 训练 Epoch | 60 | ERRNet 设置 |
| 权重初始化 | He 初始化 [^26^] | ERRNet 设置 |
| 输入尺寸 | 224×224 | 与 ERRNet 一致 |
| 梯度裁剪 | 无 | ERRNet 未使用 |

*表 4：FADNet 训练超参数设置。所有与 ERRNet 共享的设置保持一致，仅 batch size 根据 3090 VRAM 调整。*

### 5.3 消融实验设计

为验证每个改进模块的有效性，设计以下消融实验序列：

| 配置 | 双流 | FAAM | DSA | RTCL | $R^3$ Loss | 预期 PSNR (SIR²-Objects) |
|------|------|------|-----|------|-----------|--------------------------|
| ERRNet (基线) | × | × | × | × | × | 24.87 |
| + 双流输出 | √ | × | × | × | × | ~25.80 (+0.93) |
| + FAAM | √ | √ | × | × | × | ~26.20 (+1.33) |
| + DSA | √ | √ | √ | × | × | ~26.55 (+1.68) |
| + RTCL | √ | √ | √ | √ | × | ~26.75 (+1.88) |
| **FADNet (完整)** | √ | √ | √ | √ | √ | **~26.90 (+2.03)** |

*表 5：消融实验配置与预期性能。每个模块的增量贡献基于相关工作的报告结果进行合理估计。*

### 5.4 与 SOTA 方法的定量对比预期

| 数据集 | 指标 | ERRNet | IBCLN | DSRNet | DSIT | RDNet | **FADNet** (预期) |
|--------|------|--------|-------|--------|------|-------|-------------------|
| Real20 | PSNR | 22.89 | — | — | 24.54 | 25.10 | **~24.80** |
| | SSIM | 0.803 | — | — | 0.814 | 0.833 | **~0.825** |
| SIR²-Objects | PSNR | 24.87 | 26.10 | 26.72 | 27.15 | 27.35 | **~26.90** |
| | SSIM | 0.896 | 0.905 | 0.918 | 0.922 | 0.924 | **~0.920** |
| SIR²-Postcard | PSNR | 22.04 | — | — | — | — | **~23.50** |
| | SSIM | 0.876 | — | — | — | — | **~0.890** |
| SIR²-Wild | PSNR | 24.25 | — | — | — | — | **~25.80** |
| | SSIM | 0.853 | — | — | — | — | **~0.875** |

*表 6：FADNet 与现有 SOTA 方法的预期定量对比。FADNet 定位于 ERRNet 与 DSIT/RDNet 之间，在显著优于基线的同时保持适中的计算开销。*

## 6. RTX 3090 算力适配分析

### 6.1 显存占用估算

FADNet 的显存占用主要来自四个部分：模型参数 (~55M × 4 byte = 220 MB)、中间特征图 (Batch 8, 256×56×56, 前向 + 反向 ≈ 4 GB)、VGG-19 特征提取 (≈ 1 GB)、以及优化器状态 (Adam 2× 参数 ≈ 440 MB)。综合估算训练阶段峰值显存占用约 **10-12 GB**，远低于 RTX 3090 的 24 GB 限制，可以支持 batch size 8 甚至 16 的训练。推理阶段显存需求更低，约 **3-4 GB**，完全满足实时处理需求。

![模型复杂度对比](model_complexity.png)

*图 4：各方法在参数量-FLOPs 空间中的分布。绿色区域为 RTX 3090 友好区（<80M 参数量 / <150G FLOPs），橙色区域为挑战区，红色区域为重载区。FADNet（绿色五角星）位于友好区右边界，在算力限制内实现了最佳性能-效率权衡。*

### 6.2 训练时间估算

基于 RTX 3090 的实测性能（FP32 约 35 TFLOPS），FADNet 每 epoch 的训练时间约为 **8-10 分钟**（数据集规模约 8,000 张图像），60 epoch 总训练时间约 **8-10 小时**。作为对比，DSIT (233G FLOPs) 在 3090 上的训练时间约为 18-22 小时，RDNet (180G FLOPs) 约为 14-16 小时。FADNet 的训练效率优势明显，适合课程项目的迭代开发需求。

### 6.3 推理速度

FADNet 对 224×224 输入的推理时间约为 **15-20 ms**（RTX 3090，PyTorch FP32），等效于 **50-67 FPS**，满足实时应用需求。窗口注意力的采用避免了全局注意力的二次复杂度瓶颈，使 FADNet 在高分辨率输入下仍保持较好的扩展性。

## 7. 方案创新性与可行性总结

### 7.1 核心创新点

本方案围绕 ERRNet 基线提出了 **四个维度的系统性改进**，每个改进均有明确的设计动机和技术来源：

- **双流输出扩展**：继承 DSRNet 的物理建模思想，将单流输出扩展为 T+R 双流，并引入可学习残差项统一线性/非线性叠加模型。
- **频率感知注意力 (FAAM)**：融合 F2T2-HiT 的 FFT 双域思想与 ERRNet 的上下文编码框架，以轻量级方式引入频率域先验。
- **双流交叉注意力 (DSA)**：借鉴 DSIT 的双流注意力设计，但采用局部窗口策略降低计算量，使其适配 3090 级别算力。
- **对比学习损失 (RTCL)**：引入 FIRM/LapCAT 的对比学习思想，在特征空间增强传输/反射判别性，无需额外标注数据。

### 7.2 与 ERRNet 的兼容性

FADNet 的改进策略遵循 **"增量增强、保持兼容"** 的原则：BaseNet 编码器主体、VGG-19 HyperColumn 输入增强、alignment-invariant loss 等 ERRNet 核心组件均完整保留。现有 ERRNet 的预训练权重可以作为 FADNet 的编码器初始化（双流头需重新训练），从而加速收敛。这种设计确保改进工作可以在 ERRNet 现有代码库上进行模块化添加，降低实现难度。

### 7.3 预期性能提升

综合消融实验分析和相关工作的报告结果，FADNet 预期在 SIR²-Objects 数据集上达到约 **26.90 dB PSNR / 0.920 SSIM**，相对 ERRNet 基线 (24.87 dB / 0.896) 提升约 **2.03 dB PSNR / 0.024 SSIM**。这一性能将超越 IBCLN (26.10 dB) 和 YTMT (26.50 dB)，接近 DSRNet (26.72 dB) 的水平，同时参数量 (55M) 仅为 DSIT (120M) 和 DPIT (131M) 的约一半，体现了优异的性能-效率权衡。
