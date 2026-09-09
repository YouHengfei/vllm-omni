# VibeVoice-TTS 降级适配（Plan B）实现方案报告

**分支**：`support-tecoomni-vibevoice-port`（基于 `support-tecoomni` @ `dacb8e6f`，vLLM 0.25.x 基线）
**来源**：`fix/vibevoice-review-remediation`（基于较新 main，vLLM 0.27.x）
**日期**：2026-09-08
**状态**：✅ 端到端验证通过（真实 VibeVoice-1.5B 权重）

---

## 1. 背景与目标

`fix/vibevoice-review-remediation` 上的 VibeVoice 适配深度依赖 main 上的框架特性
`a2fff4f9 [Core] Add request-owned AR runner capabilities`（runner-owned
`NamedCausalKVBranch` + canonical 元数据键 + `preprocess_finalize` / `terminal_sample_drain`
等 runner hook）。`support-tecoomni` 基线不具备该框架特性。

**Plan B 目标**：在**不改动 runner 框架能力面**的前提下，把 VibeVoice 降级适配到
`support-tecoomni`，并保证将来 omni 升级到含上游 `named_kv_branch` 的版本后能**机械化迁移回上游实现**。

**核心策略**：分歧全部隔离在 `vibevoice/` 模型模块内；负 Qwen KV 分支由模型自管
（不依赖 runner-owned named KV），接口严格对齐上游 Protocol，向前迁移 = 删一个类、
接回 `bind_named_kv_branch`。

---

## 2. 方案 B 架构：模型自管负 KV 分支

### 2.1 上游 vs 方案 B

| | 上游（fix 分支） | 方案 B（本适配） |
|---|---|---|
| 负 KV 所有者 | runner（`NamedCausalKVBranch`，paged-attention） | 模型（`VibeVoiceSelfManagedNegativeKVStore`） |
| 绑定方式 | runner 调 `bind_named_kv_branch(store)` | 模型 `__init__` 自构造（`_ensure_negative_kv_branch`） |
| 注意力实现 | vLLM paged attention（`self.attn` + forward context 切换） | 复用 Qwen2 纯组件 + 模型自管 KV buffer + eager fp32-softmax SDPA |
| 元数据键 | canonical（`_omni_req_id` / `_omni_input_token_ids_cpu`） | 旧键（`request_id`），token ids 从 `input_ids` 直接取 |
| 音频转换触发 | runner 调 `preprocess_finalize` hook | `forward()` 内的 `preprocess_finalize` 兜底（runner 无该 hook） |

### 2.2 自管负分支实现要点（`negative_branch.py`）

- **接口完全对齐上游** `VibeVoiceNegativeKVBranch` Protocol：`reset_audio_segment` /
  `forward_step(request_ids, input_embeddings)` / `free`。
- **复用共享 Qwen2 权重**：`qkv_proj` / `rotary_emb` / `o_proj` / `input_layernorm` /
  `post_attention_layernorm` / `mlp` / 最终 `norm` 全部直接复用（保证权重共享、数值一致）。
- **仅替换唯一耦合点** `self.attn`（vLLM `Attention` op，绑定 paged KV + forward context）：
  改为模型自管的 per-request、per-layer K/V buffer + eager SDPA（fp32 softmax，GQA 展开）。
- **负分支与正分支 KV 完全独立**（不触碰 `self.attn`，无需 kv_cache 交换 / forward context override）。
- 负分支始终为"每请求单 token"步进（无多 token prefill），因果掩码退化为"单 query
  attend 全部历史"，无需显式 mask。

### 2.3 向前迁移路径（将来 omni 升级后）

1. 删除 `VibeVoiceSelfManagedNegativeKVStore`，恢复上游 `negative_branch.py`（`VibeVoiceNegativeBranch`）。
2. 模型侧改回 `bind_named_kv_branch(store)` + `named_kv_branch_request` 声明。
3. 元数据键改回 canonical，恢复 `preprocess_finalize` runner hook 调用。
4. `stateful.py` 的 `VibeVoiceNegativeKVBranch` Protocol 与调用点**无需改动**。

---

## 3. 代码量

### 3.1 移植自 fix 分支（非降级自研，随上游演进）

| 文件 | 行数 |
|---|---|
| `model_executor/models/vibevoice/vibevoice.py` | 1225 |
| `model_executor/models/vibevoice/diffusion.py` | 688 |
| `model_executor/models/vibevoice/stateful.py` | 653 |
| `model_executor/models/vibevoice/audio_decode.py` | 390 |
| `model_executor/models/vibevoice/processing_vibevoice.py` | 287 |
| `model_executor/models/vibevoice/runtime_config.py` | 134 |
| `model_executor/models/vibevoice/vllm_compat.py` | 63 |
| `model_executor/models/vibevoice/pipeline.py` | 54 |
| `model_executor/models/vibevoice/default_voices.py` | 26 |
| `model_executor/models/vibevoice/__init__.py` | 5 |
| `transformers_utils/configs/vibevoice.py` | 267 |
| `entrypoints/openai/tts_adapters/vibevoice.py` | 335 |
| `deploy/vibevoice.yaml` | 106 |
| **小计（移植）** | **4233** |

### 3.2 方案 B 自研降级改动（真正需要维护的部分）

| 文件 | 改动 | 说明 |
|---|---|---|
| `model_executor/models/vibevoice/negative_branch.py` | **288 行（整体重写）** | 自管负 KV store（核心，替代上游 104 行的 runner-owned wrapper） |
| `model_executor/models/vibevoice/vibevoice.py` | ~125 行 | 去 0.26+ 框架依赖、懒→急构造负分支、metadata 旧键、token ids 自取 |
| `worker/gpu_model_runner.py` | **15 行** | `_preprocess` 多模态路径传真实 `input_ids`（`preprocess_input_ids` fallback） |
| `entrypoints/openai/tts_adapters/vibevoice.py` | ~19 行 | `OutputPolicy()` 默认值、`finalize_prepared_request` 并入 `build()` |
| `entrypoints/openai/serving_speech.py` | 17 行 | `audio_chunk_semantics=delta` 增量块处理（移植自 fix 分支） |
| 注册点 ×6（registry / configs / pipeline_registry / tts_adapters / arg_utils） | ~70 行 | 机械注册 |
| **小计（自研降级，不含测试）** | **~534 行** | |
| `tests/.../test_negative_branch_self_managed_gpu.py` | 238 行 | 负分支数值对照测试（自研） |
| `tests/e2e/offline_inference/test_vibevoice_tts.py` | 153 行 | 端到端测试（自研） |
| **小计（自研测试）** | **391 行** | |

> **结论**：真正的降级自研代码约 **534 行**（核心是 288 行的自管负分支），其余 ~4200
> 行为上游移植（向前迁移时直接替换为上游版本，无需维护）。runner 共享代码仅改动
> **15 行**（`input_ids` fallback，为 main 上已验证的通用做法）。

---

## 4. 功能支持情况

### 4.1 已支持并验证

| 功能 | 状态 | 验证方式 |
|---|---|---|
| 单阶段 AR TTS 生成（参考音频 voice cloning） | ✅ | 端到端（真实权重） |
| CFG 负分支（自管，与上游数值一致） | ✅ | conformance 测试（max_abs_diff ≤ 0.031，tol 0.04） |
| CFG diffusion（正负条件 guidance） | ✅ | 端到端 |
| 声学解码 + 语义反馈回路 | ✅ | 端到端 |
| Delta 波形增量输出（3200 采样/音频 token） | ✅ | 端到端 + serving_speech delta 语义 |
| 多请求连续 serving（请求间状态隔离/清理） | ✅ | 4 个连续请求全部成功 |
| `allowed_token_ids` token gate（audio BOS/EOS/pad/im_end） | ✅ | 端到端 |
| preprocess 元数据（旧键 `request_id`/`_omni_is_prefill`/`_omni_num_computed_tokens`/`_omni_prompt_len`） | ✅ | 端到端 |
| `forward()` 内 `preprocess_finalize` 兜底 | ✅ | 端到端 |
| Microsoft 原始 config schema 归一化（`decoder_config`→`text_config` 等） | ✅ | 真实权重加载成功 |
| tokenizer 契约解析（`preprocessor_config.json` → Qwen2.5-1.5B） | ✅ | 端到端 |

### 4.2 功能取舍（相对 fix 分支 / 上游）

| 功能 | 状态 | 影响 |
|---|---|---|
| `drain_terminal_sampled_tokens`（硬上限采样排空） | ⚠️ 不可用 | runner 无该 hook；达到 max_tokens 时按正常停止（不影响正确性，仅少一个优化） |
| `expose_finish_reason`（HTTP/SSE finish_reason 元数据） | ⚠️ 不可用 | serving_speech 无配套管线；音频生成不受影响 |
| diffusion / decode 侧 CUDA graph | ⚠️ 测试配置关闭 | 可开启；`forward()` 非纯调用使 AR graph 本就关闭（`enforce_eager=true`），侧 graph 独立可开 |
| `preprocess_finalize` runner hook | ⚠️ 走 forward 兜底 | 功能等价，仅 forward 无法被 vLLM 捕获为 FULL decode graph（VibeVoice 默认 eager，无实际损失） |
| `_omni_input_token_ids_cpu` CPU 镜像 | ⚠️ 改为从 GPU `input_ids` 读取 | 每步一次小 D2H 同步；eager 模式下开销可忽略 |
| 在线 serving（`/v1/audio/speech`） | ✅ 已验证 | 需修复 `_TTS_MODEL_STAGES` 注册（见 §7）+ `NO_PROXY` 绕过代理 |

### 4.3 在线 serving

| 项 | 结果 |
|---|---|
| `POST /v1/audio/speech`（参考音频 voice cloning） | ✅ 200，返回 24kHz WAV（3.87s，finite，非静音） |
| 前置修复 | vibevoice 加入 `_TTS_MODEL_STAGES`（否则 `_find_tts_stage` 返回 None → 走 raw-text 兜底） |
| 环境 | 必须 `NO_PROXY=127.0.0.1,localhost` 绕过代理（否则 localhost 请求被代理拦截返回空 503） |

### 4.4 环境要求（实测发现）

- `transformers >= 5.10.1`（config 依赖 `vibevoice_acoustic_tokenizer`；实测环境 5.16.1）。
  support-tecoomni 现 pin `>=5.5.3`，**需提升下界**。
- `VLLM_USE_FLASHINFER_SAMPLER=0`：本环境缺 `ninja`，flashinfer JIT 编译失败；回退原生采样
  **不影响** `allowed_token_ids` token gate 正确性。（若装 ninja 则无需此开关。）
- tokenizer：`Qwen/Qwen2.5-1.5B`（经 preprocessor_config 契约解析；离线环境需预缓存或 `HF_HUB_OFFLINE=1`）。

---

## 5. 性能情况（H100 80GB，真实 VibeVoice-1.5B，bf16，TP=1，enforce_eager）

| 指标 | 数值 | 说明 |
|---|---|---|
| 引擎初始化 | ~50–70 s | 含权重加载（~5.4GB safetensors）+ KV 池 + 子模块构造 |
| **TTFT**（首音频块） | **~1.2 s** | 流式首 chunk |
| **RTF（实时率）** | **0.30–0.35（非流式）/ ~0.73（流式）** | RTF<1 即快于实时；流式含 per-chunk 输出消息开销 |
| **吞吐** | **~21–25 音频 token/s（非流式）** | 1 音频 token = 3200 采样 = 0.133s 音频 |
| 长文本（147 token / 19.6s 音频） | RTF 0.30，24.8 tok/s | RTF 随长度稳定 |
| 负 KV buffer 扩展性 | 147 token 无退化 | `torch.cat` 增长在典型 TTS 长度（数十~数百 token）无可见 O(n²) 影响 |

### 5.1 侧 CUDA graph（diffusion + decode）A/B

侧 graph 经**懒捕获**生效（首次使用某 batch key 时 capture），**无需 runner 改动**——只需
`vibevoice_runtime_config.diffusion_cuda_graph/decode_cuda_graph: true`（shipped `vibevoice.yaml` 默认已开）。

稳态 RTF（warmup 后 3 次平均，长文本非流式）：

| 配置 | RTF | 吞吐 |
|---|---|---|
| eager（侧 graph 关） | **0.304** | ~25 tok/s |
| 侧 graph 开 | **0.189** | ~40 tok/s |

→ 侧 graph 带来 **~1.6× RTF 提速**（RTF 降 ~38%），与上游 "~50% RTF 收益" 同量级。

> **性能结论**：RTF 0.30–0.35（非流式）意味着生成速度约为实时的 **3 倍**，单请求 TTS
> 完全实用。流式 RTF ~0.73 仍快于实时。**自管负分支未引入可测的性能退化**（CFG 本身
> 使每步需跑正+负两次 Qwen，这是模型固有成本，非方案 B 引入）。

---

## 6. 验证情况汇总

| 验证 | 结果 |
|---|---|
| 负分支数值对照（vs 独立 HF Qwen2 参考） | ✅ 单请求 max_abs_diff=0.03125、批处理错位=0.02344（tol 0.04），跨页、批处理、reset 均正确 |
| 模型构造（tiny config + dummy 权重） | ✅ 全部子模块（audio_tower/semantic/diffusion/decode）构造成功 |
| 端到端（真实 VibeVoice-1.5B，24kHz 波形） | ✅ PASS：`(89600,)` float32 @ 24kHz = 3.73s 音频，finite，%3200=0 |
| 音频质量 | ✅ 真实语音：频谱质心 1796Hz、动态范围 74dB、过零率 0.114（均语音区间） |
| 多请求连续 serving | ✅ 4/4 连续请求成功，无状态泄漏 |
| 全模块导入 / 注册完整性 | ✅ pipeline/model/config/adapter 注册全部成功 |
| 负分支 conformance 回归（eager init 改动后） | ✅ 仍通过 |
| 在线 serving（`/v1/audio/speech`） | ✅ PASS：200，24kHz WAV 3.87s，finite，非静音 |
| 侧 graph A/B（eager vs graph） | ✅ RTF 0.304 → 0.189（~1.6× 提速），懒捕获生效 |

---

## 7. 端到端过程中发现并修复的兼容性问题

1. **`input_ids=None`（多模态路径）**：support-tecoomni `_preprocess` 在模型有多模态
   输入时把 `input_ids=None` 传给 `preprocess`，但 vibevoice 需要 token ids 检测控制
   token。**修复**：`_preprocess` 加 `preprocess_input_ids` fallback（15 行，fix 分支已验证的通用做法）。
2. **负分支 reset 丢失**：负分支原为懒构造（forward 才建），但 `start_audio_segment`→
   `reset_audio_segment` 在 preprocess（forward 之前）执行，导致首次 reset 被跳过、
   `forward_step` 报 "must be reset"。**修复**：store 改为 `__init__` 急构造（构造不分配 GPU，安全）。
3. **`OutputPolicy(expose_finish_reason=...)`**：该参数及其 serving 管线 support-tecoomni
   不存在。**修复**：adapter 用默认 `OutputPolicy()`。
4. **`finalize_prepared_request` hook**：support-tecoomni orchestrator 不调该 hook。
   **修复**：其逻辑（`prompt_token_ids` + `multi_modal_uuids`）并入 `build()`，hook 保持幂等以兼容两分支。
5. **环境**：缺 `ninja`（flashinfer JIT）→ `VLLM_USE_FLASHINFER_SAMPLER=0`；tokenizer 网络
   瞬时失败 → `HF_HUB_OFFLINE=1`。
6. **在线 serving：vibevoice 未注册为 TTS stage**（`fb567460`）：`_find_tts_stage` 依据
   `model_stage in _TTS_MODEL_STAGES`，vibevoice 的 `model_stage="vibevoice"` 缺失 →
   `_tts_stage=None` → adapter 不被解析 → 走 raw-text 兜底（prompt 不渲染、无 audio_bos、
   负分支饿死）。修复：加 `_VIBEVOICE_TTS_MODEL_STAGES` + `_detect_tts_model_type` 分支。
7. **代理拦截 localhost**：环境设了 `http_proxy`，对 `127.0.0.1` 的请求被代理拦截返回空
   503。在线测试/客户端必须设 `NO_PROXY=127.0.0.1,localhost`（fix 分支在线测试也这么做）。

> **更正此前的“流式崩溃”记录**：经查证那不是 bug——`generate(py_generator=True)` 的
> `_run_generation_with_generator` 在 finally 中 `self.close()`，流式生成器被消费完即关闭
> 引擎，是 Omni offline API 的模型无关行为（连续两次流式 generate 需重建引擎）。在线 serving
> 不反复调 `generate(py_generator=True)`，不受影响。

---

## 8. 已知限制与后续工作

| 项 | 说明 | 建议 |
|---|---|---|
| ~~流式（py_generator）路径偶发崩溃~~ | **非 bug**：`generate(py_generator=True)` 设计上消费完即关引擎（Omni offline API 行为） | 无需修复；在线 serving 不受影响 |
| 负 KV buffer `torch.cat` O(n²) | 极长生成（数千 token）可能有拷贝开销；典型 TTS（数百 token）无影响 | 如需超长生成，改 chunked 预分配（capacity 倍增） |
| 依赖版本 | `requirements` 需 `transformers>=5.10.1`、`diffusers` 0.38→0.40 | 评估对 support-tecoomni 其他模型影响后提升 |
| 侧 CUDA graph | ✅ 已验证生效（懒捕获，~1.6× RTF）；shipped `vibevoice.yaml` 默认已开 | 无 |
| 在线 serving | ✅ 已验证（需 `_TTS_MODEL_STAGES` 修复 + `NO_PROXY`） | 无 |

---

## 9. 提交历史（`support-tecoomni-vibevoice-port`）

```
5a1b8109 [Test] VibeVoice online serving E2E on support-tecoomni
fb567460 [Frontend] VibeVoice: register as TTS model stage for online serving
293d0a9a [Doc] VibeVoice Plan-B support-tecoomni implementation report
9e67942e [Test] VibeVoice offline E2E on support-tecoomni (real checkpoint)
76c0f1dc [Bugfix] VibeVoice E2E on support-tecoomni: input_ids in preprocess + eager negative branch
e73a6b9a [Frontend] VibeVoice adapter: downgrade to support-tecoomni TTS base API
0806105b [Model] VibeVoice: register pipeline/model/config/adapter on support-tecoomni
7b4ed5a2 [Model] VibeVoice: wire self-managed negative branch, drop 0.26+ framework deps
cce36c36 [Model] VibeVoice: self-managed negative KV branch (Plan-B downgrade)
```
