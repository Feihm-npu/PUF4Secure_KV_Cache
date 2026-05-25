**总体判断**
实验结果是正向的：当前 PoC 已经证明“PUF-derived basis + K/V 共同保护”可以阻断 Qwen3-0.6B 上的 inversion、collision、injection，并且同设备恢复保持可用，异设备恢复接近随机。但现在还不能完全锁定 proposal 的最终实现，因为目前主要是 **cache-level transform + decloak**，还不是完整的 **attention-equivalent model-integrated basis**。

**关键结论**
1. **必须保护 K+V，V-only 不够稳**
- Plain baseline 很强：
  - V inversion: `1.00`
  - collision V@L0: `0.740`
  - injection ROUGE-L: `0.278`
- 所有 V-only 方案都能杀死 V inversion。
- 但 V-only injection 仍有 `0.057–0.093` ROUGE-L，且 K 仍在原始 basis，模型仍可能 coherent attend 到 prefix。
- 因此 production 方案不应采用 V-only。

2. **K+V Givens 是当前最强工程候选**
从 aggregate 看：

| Variant | inv | collision | injection | protect | recover |
| --- | ---: | ---: | ---: | ---: | ---: |
| P3 KV Givens | 0.00 | 0.00 | 0.00 | 0.099s | 0.100s |
| P3 KV Hadamard | 0.00 | 0.075 | 0.00 | 0.564s | 0.591s |
| P3 Givens+Hadamard | 0.00 | 0.075 | 0.00 | 0.597s | 0.378s |
| P4 G+H row | 0.00 | 0.00 | 0.00 | 0.386s | 0.431s |

目前数据更支持 **K+V Givens**，而不是马上锁定 `K=Givens, V=Hadamard`。Hadamard 的安全收益在这批攻击中没有明显超过 Givens，但性能差很多。

下一轮优先测：
- `P4_KV_givens_row`
- `P4_KV_givens_block`
- `P5_KV_givens_session_refresh`
- `P5_KV_givens_row_session_refresh`

1. **row layout / session refresh 是 profiling 防御的核心**
Profiling 结果非常重要：

| Scenario | in-sample residual | transfer residual |
| --- | ---: | ---: |
| S1 fixed no-layout same-session | `0.001` | `0.001` |
| S2 fixed row-layout same-session | `0.667` | `0.667` |
| S3 fixed no-layout session-refresh | `0.001` | `1.490` |

解释：
- 固定 basis 且无 layout 时，chosen-input profiling 能拟合出一个可用变换。
- `basis_recovery_err ≈ 1.09` 不代表攻击失败，因为当前 rows 少于 head_dim，Procrustes 不唯一；真正要看 residual。
- row layout 直接破坏 row alignment，显著提高 residual。
- session refresh 使同 session 可拟合，但跨 session 不迁移。

所以最终方案必须包含：
- session nonce
- row/block layout
- 最好 per-session + per-block/context derivation

4. **PUF noise 结果支持 fuzzy extractor，但当前 DBM 指标需要重定义**
noise 结果显示：
- BER `0–0.2` 时 fidelity 不变，说明 fuzzy correction capacity 内恢复稳定。
- BER `0.3+` 时 same-device 恢复崩溃，K/V rel-L2 接近随机。
- 这符合预期。

但 `noise_summary.json` 里的 `DBM` 用的是 leakage ROUGE-L：
- same-device legitimate injection 本来就应该恢复 plaintext，所以 leakage 分数高是正常 utility。
- wrong-device leakage 低是安全性。
- 因此当前 `DBM = wrong_leak - noisy_leak` 得到负值，不应解释为 device binding 差。

建议把 DBM 改成距离型：
- `D_same = KL/plain-vs-same-device` 或 `1 - output agreement`
- `D_wrong = KL/plain-vs-wrong-device` 或 `1 - output agreement`
- `DBM = D_wrong - D_same`

5. **当前 PoC 仍是 cache-level defense，不是最终 attention-basis defense**
目前同设备路径是：
- protect cache
- recover/decloak cache
- 再 decode

这证明了：
- protected cache 离开设备后不可用
- 同设备可恢复
- wrong device 无法恢复

但 proposal 更强的主张是：
- 合法设备在 PUF attention basis 中直接继续推理
- cache 本身就是 model-native 的 device-bound intermediate state

因此下一步必须做 Level-3：
- post-RoPE Q/K Givens wrapper
- V basis + output inverse wrapper
- 验证不 decloak 也能 same-device decode

**需要修正或补充的实验**
1. 加 secret-specific leakage metric
- ROUGE-L 和 char-F1 对安全结论不够精准。
- 建议针对 synthetic prompts 加：
  - exact secret presence: `482913`, `blue-river`, `555-0108`
  - digit leakage rate
  - codename leakage boolean
  - PII substring match

2. profiling 要加强
当前只有 54 rows，远小于 `head_dim=128`，Procrustes 是欠定问题。下一轮应加：
- 1k prompts
- rows > 128, > 512, > 4096
- unknown permutation recovery attack
- block-level assignment attack
- profiling 后再跑 inversion/collision/injection，而不只看 residual

3. 不要过早锁定 Hadamard
当前 `P3 KV Givens` 是最优速度/安全候选。Hadamard 可以保留为 stronger-mixing baseline，但不是默认主线，除非大规模 profiling 证明 Givens 更容易被恢复。

**推荐下一步**
优先做 4 件事：

1. 实现并测试 `KV-Givens + row/block layout + session refresh`。
2. 增加 secret exact-match leakage 指标。
3. 扩大 profiling 到 ≥1k prompts，并加入 permutation-alignment 攻击。
4. 实现 post-RoPE attention wrapper，证明“不 decloak 也能 same-device decode”。

**当前最合理候选**
```text
Q/K path: per-layer, per-kv-head post-RoPE Givens basis
V/O path: 先用 Givens，Hadamard 作为增强候选
Layout: session-refreshed row/block permutation
PUF: fuzzy-extractor-stable root + HMAC context derivation
```
