# 论文角度的进展盘点与待办（comments.md）

面向顶会（CCS / S&P / USENIX Security / NDSS）的"可发表性"评估。
基线见 `docs/PUF_BASIS_RESULTS.md`、`docs/PUF_PLAN_v1_RESULTS.md`、
`docs/REPRODUCTION_RESULTS.md`。完整实验流水见 `docs/EXPERIMENT_PROGRESS.md`
（Wave 0–11）。论文草稿见 `paper_latex/`。

> 状态图例：✅ 已完成　⚠️ 部分完成（含剩余项）　❌ 未做

## 〇、自上版以来的关键变化（务必先读）

上一版 comments.md 写于"只有 Qwen3 + 3 条 prompt"的阶段，现已完成 Wave 0–11
的大规模硬化。**最重要的新发现（上版没有）**：

- **正交基不是完整的机密性机制（Wave-9 负面结果）**。正交变换保范数
  `‖XO‖₂=‖X‖₂`，cache-only 攻击者用候选 secret 的范数签名直接匹配，
  在 Qwen3 / Llama / Qwen2.5 上 top1/MRR=1.000 重新破解。这把论文叙事从
  "正交基=机密"改写成"正交基=device/session 绑定基线 + 必须移除 basis 不变量"。
- **affine-mask 重设计（Wave-11）**作为 no-sidecar 缓解：存 `XO_s + M_{s,i}`，
  在 attention 内重生成并减去 PUF 掩码。std128 把 norm-L2 top1 从 1.000 降到
  Qwen3 0.050 / Llama 0.000。**仍是 prototype**，缺 utility / long-decode /
  performance / 更强攻击审计。

新叙事骨架："攻击复现 → 正交基绑定（修复 inversion/injection）→ 定位固定
basis 缺陷（session refresh）→ **定位 norm 侧信道（affine mask 重设计）**"。

## 〇bis、Wave 12-13（2026-06-09，自动托管模式）— review insights 已实施

针对上一轮 review 的 7 条 insight，已全部落地（详见 `docs/EXPERIMENT_PROGRESS.md`
"Waves 12-13" 与已重编译的 `paper_latex/`）：

1. ✅ **重定位贡献为物理 non-migratability**（非"免解密"）。新增 migration 对照实验
   （E3）：软件密钥随 cache 迁移即被攻破（V-inv 1.0），PUF-Cache 即便攻击者拥有完整
   软件栈、换设备仍不可解（V-inv 0.0，relL2≈√2）。abstract/intro/discussion/conclusion 已改写。
2. ✅ **norm 泛化为整类 basis 不变量**。新增 gram_l2 / svd_l2 攻击：正交 cache 在
   Gram 矩阵和奇异值谱上同样 top1=1.0（Qwen3+Llama）；affine-mask 把整类压到 ~chance
   （0.033-0.067）。sec3 设计原理 + sec4 审计表已泛化。
3. ✅ **flash/paged 兼容性**。正交路径跑通 `scaled_dot_product_attention`（max diff 3.96e-5）；
   sec3/sec5 增加 fused-kernel 讨论；affine 路径需 kernel 级 de-mask（future）。
4. ✅ **收紧 niche**（边缘/无 TEE/物理绑定）。sec5 新增 "Scope" 子节。
5. ✅ **std 权衡曲线 + O(n) affine mask**。修复了 affine mask 的 O(n²) 性能 bug
   （L1024 6 小时→12 秒）；std 4→256 的 security/precision 曲线已成表（sec4 tab:std-sweep）。
6. ✅ **inversion 降级为单模型**。RQ1 + Limitations 明确 inversion 仅 Qwen3 square v_proj，
   跨模型证据由 injection + candidate 承担。
7. ✅ **性能 rerun（长 horizon）**。decode128×8，方差脱噪：正交 Qwen3 -15%、7B -18%、
   affine -32%。MMLU（标准基准）补上，plain==wrapped Δ0。

下一轮真正剩余的硬骨头：fp16 FlashAttention 的数值稳健性 + fused affine-mask kernel；
真实 PUF 硬件标定；真实私有语料（Enron/LongBench）。

## 〇ter、Wave 14（2026-06-09）— 残余 future work 已尽量推进

- ✅ **fp16 数值稳健性**:fp16 仅 15.6% 发散(bf16 是 56.25%,fp32 是 0%)。fp16(10 位
  尾数)比 bf16(7 位)稳 ~3.6×;FlashAttention 用 fp16,所以低精度部署比 bf16 结果暗示的更可行。
- ✅ **flash 兼容的 affine 路径**:affine de-mask 是 kernel 前的逐元素操作,affine+SDPA
  已跑通(max diff 4.0e-4, token match 1.0)。剩"把减法 fuse 进 kernel cache-load 阶段"。
- ✅ **真实长上下文 utility**:PG-19 @ 4096 token,plain vs wrapped PPL delta +1.4e-6
  —— wrapper 在长序列上不累积误差。
- ✅ **真实语料泄露**:Enron 真实邮件 200 条,plain 7% 泄露、protected 0%。
- ⛔ **真实 PUF 硬件**:需 FPGA PUF 流 + fuzzy-extractor 标定,本环境无硬件,诚实列为 limitation,未造假。
- ⏳ 仍未做(需新硬件/工程):fused affine-mask kernel;真实私有记录里挖掘 secret(当前仍是 seeded)。

## 一、当前已有的论文素材（资产盘点）

| 论文模块 | 已具备的证据 | 强度 |
|---|---|---|
| 威胁模型/攻击复现 | NDSS KV-cache 攻击（V-inversion top1=1.00、injection 逐字泄露 secret）已复现 | 强 |
| 核心防御 | 三层设计：cache-level Givens / layout / session-refresh / Level-3 attention wrapper | 强 |
| 攻击下的安全性 | 1000-prompt Procrustes profiling（固定 basis 必破、layout 单用部分被破、session-refresh 鲁棒 0.000）；KPA；native Hungarian | 很强 |
| 等价性证明 | wrap_fp ≡ plain_fp 逐 token 完全一致，5 模型跨族 | 强：可上升为定理 |
| 数值现实性 | bf16 漂移定位为量化噪声（56.25% 发散曲线 vs fp32 0%） | 强：诚实且深入 |
| **norm 侧信道（新）** | 正交基保范数 → 候选匹配 top1=1.000 三模型；affine-mask 缓解到近 chance | 很强：最诚实的自我审计 |
| 多模型泛化 | Qwen3-0.6B / Llama-3.2-1B / Qwen2.5-1.5B / Llama-2-7B / Qwen2.5-7B | 强 |
| 真实语境隐私 | CC-News seeded PII 300-prompt，plain 23% → protected 0% | 中强 |
| 性能 | 端到端 prefill/decode latency、tokens/s、KV 显存（5 模型） | 中 |
| 模糊 PUF | BER≤0.20 utility 保持、≥0.30 崩溃 | 中 |

## 二、从论文标准看的待办（按重要性排序）

### P0 — 不补就会被拒的硬伤

1. **评测规模** —— ⚠️ 大部分完成
   - 多模型：✅ 已加 Llama-3.x、Qwen2.5、7B 规模，GQA/MHA/不同 head_dim 都成立。
   - 真实数据集：⚠️ 已有 500/120-prompt 合成 PII benchmark（Wilson CI）+ 300-prompt
     CC-News real-context seeded。**剩余**：仍无标准隐私/长文本语料
     （Enron / 标准 PII benchmark / LongBench）上的泄露率，审稿人可能要求"真实
     私有数据"而非"真实语境 + 注入合成 secret"。

2. **效用用标准基准量化 + 解决 bf16 漂移** —— ⚠️ 大部分完成
   - fp32-attention-activations：✅ 已实现，5 模型逐 token 等价、PPL/HellaSwag Δ≈0。
   - bf16 漂移：✅ 已定位为量化噪声，给出 56.25% 发散曲线（非协议错误）。
   - 标准基准：⚠️ 目前仅 AG News PPL + HellaSwag（128 样本，paper 自述"非完整
     基准套件"）。**剩余**：MMLU / LongBench 等更主流的 utility 点，提升说服力。

3. **形式化威胁模型与安全论证** —— ✅ 基本完成
   - ✅ A0–A3 攻击者分类、能力边界（见 `SECURITY_ARGUMENT_AND_COMPARISONS.md`）。
   - ✅ Procrustes reduction（同会话恢复 O_s ≡ 正交 Procrustes 问题）+ 范数不变量论证。
   - ⚠️ **剩余（可选加分）**：把经验论证上升为信息论/归约定理（"无 O 时恢复 K/V
     等价于某难题"目前是 sketch 级，非严格 reduction）。

### P1 — 决定论文档次（从 borderline 到 accept）

4. **更强的自适应攻击** —— ✅ 完成
   - ✅ sort-by-norm 升级为 Hungarian / block-aware Hungarian。
   - ✅ known-plaintext 攻击（`run_kpa_native.py`，4/16/64 known prompts）。
   - ✅ Level-3 native cache 上 1000-prompt profiling（`profiling_native_summary.json`）。

5. **系统开销（performance）** —— ⚠️ 部分完成
   - ✅ 端到端 prefill/decode latency、tokens/s、KV 显存（Qwen3 + Qwen2.5-7B）。
   - ⚠️ **剩余**：GPU 向量化 Givens 未跑赢 dense einsum；Triton standalone kernel 也
     不占优。可信的"优化后开销"需把旋转/掩码 **fused 进 attention/QKV kernel**，
     目前仍是 future work。affine-mask 路径的性能尚未测。

6. **与替代方案的对比** —— ✅ 完成
   - ✅ 对比表已建（No defense / AES / TEE / software random basis / PUF-Cache /
     unit-norm sidecar / affine-mask），含安全假设、开销、是否需 decrypt-before-attend。
     见 `SECURITY_ARGUMENT_AND_COMPARISONS.md`。待 cite 补全后并入 Related Work。

### P2 — 加分项 / 提升可信度

7. **真实 PUF 集成** —— ❌ 未做（仍 `PUFSim`）
   - 哪怕接一次真实 FPGA PUF 流做 fuzzy-extractor 标定，也能把"模拟"升级为
     "硬件验证"。论文目前明确把它列为 limitation / future work。

8. **密钥生命周期 / 多租户** —— ❌ 大部分未做
   - HMAC-SHA256 context derivation 已实现并在 Discussion 提及 session nonce，
     但多用户隔离、重放/侧信道、密钥轮换的系统化讨论仍缺。
   - ⚠️ **新增子项**：affine-mask 引入的 sidecar/mask 信任边界需要在威胁模型里
     说清（sidecar 是否被导出/迁移/dump）。

9. **消融实验完整化** —— ✅ 基本完成
   - ✅ Level-3 native + Hungarian 那一格已补；每层防御单独 vs 组合矩阵基本成型。
   - ⚠️ **剩余**：affine-mask 与 orthogonal-only 的并排消融（utility×安全）还没补全。

## 三、建议的下一步执行顺序

上版列的两件（fp32-utility、Level-3 native Hungarian profiling）**均已完成**。
当前最该做的（按 ROI 排序）：

1. **affine-mask 路径补齐审计**（当前最大的"半成品"风险）：
   - utility（PPL/HellaSwag/long-decode）、性能开销、更强的 sidecar/mask-aware
     候选攻击。这是论文从"发现问题"走向"给出可信修复"的关键一格。
2. **标准 utility 基准扩充**（MMLU / LongBench）——拆掉"只在 AG News+HellaSwag
   上测"的审稿质疑。
3. **真实隐私语料**（Enron / 标准 PII benchmark）——补 P0-1 的最后一块。
4. （加分）信息论/归约定理化安全论证；真实 PUF 标定；多租户/密钥生命周期讨论。

## 四、提交前的收尾清单（来自各 doc + paper）

- [ ] Llama-2-7B 完整 128-sample utility（若需第二个 7B utility 点）。
- [ ] PII 从 500 扩到 1,000（若要计划区间上端）；Llama 仅 120。
- [ ] affine-mask 的 utility / long-decode / performance / 更强攻击审计。
- [ ] `reference.bib` 与 Related Work 的 `\needcite`（PagedAttention/vLLM、PUF/
      fuzzy extractor、TEE/confidential GPU、额外 KV-cache 泄露）补全。
- [ ] 把 A0–A3 与对比表正式并入 paper 的 threat model / discussion。
