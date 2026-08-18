
# VibeVoice-TTS 官方 checkpoint 适配说明

本文记录非 Realtime VibeVoice-TTS 在 vLLM-Omni 中的配置契约。适配范围严格限于 Microsoft VibeVoice-1.5B TTS checkpoint 的推理；VibeVoice 其他参数规模、Realtime、ASR 和训练均不在范围内。公开输入是 Microsoft 官方原始 checkpoint；运行时统一使用 Transformers PR #40546 的 HF schema，以便复用已合入的 Acoustic Tokenizer 和参考完整 TTS PR。

## 1. 输入与运行时边界

公开输入：

```text
microsoft/VibeVoice-1.5B
```

官方 `config.json` 使用：

```text
acoustic_tokenizer_config
semantic_tokenizer_config
decoder_config
diffusion_head_config
```

`VibeVoiceConfig` 在纯 Config 构造阶段将其转换成 HF runtime schema：

```text
VibeVoiceConfig
|- audio_config
|  `- transformers.VibeVoiceAcousticTokenizerConfig
|- semantic_model_config
|  `- transformers.VibeVoiceAcousticTokenizerEncoderConfig
`- text_config
   `- transformers.Qwen2Config
```

Diffusion 参数转换后保存在顶层，与 PR #40546 的模型消费方式一致。Config 转换不实例化模型、不读取权重、不初始化 CUDA。

经过 PR #40546 离线转换的 HF checkpoint 也可直接加载；两种输入最终得到同一种 runtime Config。

## 2. Config 转换

### 2.1 Acoustic/Semantic

```text
vae_dim                 -> hidden_size
encoder_n_filters       -> num_filters
reverse(encoder_ratios) -> downsampling_ratios
split(encoder_depths)   -> depths
layernorm_eps           -> rms_norm_eps
weight_init_value       -> initializer_range（同时保留原元数据）
```

Acoustic VAE 还执行：

```text
vae_std = fix_std / 0.8
```

目标 1.5B 中为 `0.5 / 0.8 = 0.625`。Semantic Encoder 不采样 Acoustic VAE noise，使用上游 Encoder Config 默认值。

### 2.2 Qwen2

```text
decoder_config -> text_config
torch_dtype     -> dtype
```

运行时 `get_text_config()` 返回 `Qwen2Config`，供 vLLM 读取：

```text
hidden_size = 1536
num_hidden_layers = 28
num_attention_heads = 12
num_key_value_heads = 2
max_position_embeddings = 65536
```

### 2.3 Diffusion

官方 `diffusion_head_config` 被摊平到顶层：

```text
head_layers                         -> num_head_layers
head_ffn_ratio * hidden_size        -> intermediate_size
prediction_type                     -> prediction_type
ddpm_num_steps                      -> ddpm_num_steps
ddpm_num_inference_steps            -> ddpm_num_inference_steps
cosine                              -> squaredcos_cap_v2
```

不创建独立 `diffusion_head_config` runtime 子对象，避免与顶层字段形成两个配置来源。

## 3. PR Diffusion Head 如何消费 Config

PR #40546 的 Diffusion Head 构造方式是：

```python
self.diffusion_head = VibeVoiceDiffusionHead(config)
```

它直接读取顶层：

```text
config.hidden_size                  # property -> text_config.hidden_size
config.audio_config.hidden_size     # latent size
config.num_head_layers
config.intermediate_size
config.rms_norm_eps
config.hidden_act
config.frequency_embedding_size
config.diffusion_max_period
config.mlp_bias
```

上述 DDPM 字段按 HF checkpoint schema 保留在顶层。PR 的推理循环实际优先从 `generation_config.json` 读取 `noise_scheduler_class`、`noise_scheduler_config` 和 `num_diffusion_steps`；Omni 可用顶层 DDPM 字段作为构造 scheduler 的模型默认值，但请求级生成参数仍应覆盖它们。

因此不能为了 logical stage 再添加重复的 Diffusion 子 Config。若 Omni 需要类型化 runtime view，应在 `model_executor/models/vibevoice/runtime_config.py` 中从顶层生成不可序列化的 runtime dataclass。

## 4. Transformers 与 Omni 的职责

Transformers 提供：

- `AutoConfig` / `PreTrainedConfig`；
- `Qwen2Config`；
- `VibeVoiceAcousticTokenizerConfig`；
- `VibeVoiceAcousticTokenizerEncoderConfig`；
- Acoustic Encoder/Decoder 模型；
- HF Processor/完整 TTS PR 作为复用或正确性参考。

Omni/vLLM 负责：

- `VibeVoiceForConditionalGeneration` ModelRegistry；
- vLLM Qwen2、PagedAttention、KV Cache、TP 和 batching；
- logical stages；
- 官方权重名到 HF/Omni runtime 参数名的映射；
- diffusion execution 和同一 AR stage 内的 Acoustic Decoder；
- Processor 接入、stage connector 和部署参数。

## 5. Config 注册

当前 Transformers 已有：

```text
qwen2
vibevoice_acoustic_tokenizer
vibevoice_acoustic_tokenizer_encoder
```

Omni 只补充顶层：

```text
vibevoice -> vllm_omni...VibeVoiceConfig
```

不会覆盖未来 Transformers 内置的顶层 VibeVoice Config。vLLM Config registry 也注册该顶层，使前端和 EngineCore 都能调用：

```python
get_config("microsoft/VibeVoice-1.5B", trust_remote_code=False)
```

## 6. 后续模型结构与 stage 决策

当前目录：

```text
vllm_omni/model_executor/models/vibevoice/
|- vibevoice.py              # 模型组装、MM embedding 和权重映射/加载
|- diffusion.py              # 有权重 Diffusion Head + 无参数 CFG/DPM sampler
|- audio_decode.py           # Acoustic Decoder + Semantic feedback 单-token kernel
|- stateful.py               # request-local M4c 状态机；不拥有 Qwen KV
|- processing_vibevoice.py   # stateless MM Processor
|- vllm_compat.py            # 私有 vLLM API 边界
|- pipeline.py
`- __init__.py
```

M4a 后 Diffusion 职责已形成独立模型侧模块：Head 只执行单 timestep prediction，sampler
负责多步 CFG/DPM 数值求解；二者位于同一个 `diffusion.py`，但参数和状态职责不重叠。
权重映射仍保留在拥有 `load_weights()` 的 `vibevoice.py`。M4c request-local 状态已拆到
`stateful.py`，模型类只保留 Omni hooks 和模块组装；不为 VibeVoice 增加独立 runtime
scheduler。

VibeVoice 更适合先实现为一个 `LLM_AR` logical stage，拓扑与 VoxCPM2 类似：

```text
Stage 0（final audio output）
|- Acoustic Encoder（reference prompt）
|- Semantic Encoder（生成 waveform 的逐步反馈）
|- vLLM Qwen2 positive/negative CFG branches
|- Multi-modal projectors
|- Diffusion Head（每个 AR audio step 内执行若干 denoise steps）
`- stateful Acoustic Decoder（立即将 latent 解码为 waveform chunk）
```

PR #40546 的 generation loop 在每个 AR step 内连续执行：

```text
Qwen hidden state
-> positive/negative CFG condition
-> diffusion denoise loop
-> latent scaling/bias inverse
-> acoustic decoder with causal padding cache
-> waveform chunk
-> latent embedding 反馈给下一次 AR step
```

这些操作共享同一个请求的 KV cache、negative CFG cache、Acoustic Decoder padding cache 和下一步 embedding，拆成 Code2Wav stage 会增加跨 stage 状态同步和逐 token connector 开销，当前没有独立扩缩容收益。因此第一版不要拆分第二个 stage。只有后续 profiling 证明 Acoustic Decoder 是可独立批处理的瓶颈，且能明确传输和恢复 causal decoder cache 时，才重新评估两 stage。

Pipeline 建议参考：

```text
vllm_omni/model_executor/models/voxcpm2/pipeline.py
```

当前 `vllm_omni/model_executor/models/vibevoice/pipeline.py` 已固定为：

```text
execution_type = LLM_AR
final_output = true
final_output_type = audio
engine_output_type = audio
owns_tokenizer = true
requires_multimodal_data = true
```

模型构造可直接使用：

```python
config.audio_config.encoder_config
config.audio_config.decoder_config
config.semantic_model_config
config.text_config
```

## 7. 官方权重兼容

Config 转换只统一静态配置，不会重命名 safetensors 中的 parameter key。官方权重必须在 Omni `load_weights()` 中执行一层在线映射；用户不需要提前生成 HF 权重副本。

当前在 `vibevoice.py` 中根据 normalized Config 构造标准 vLLM mapper：

```python
mapper = _build_vibevoice_weights_mapper(self.config)
AutoWeightsLoader(self).load_weights(weights, mapper=mapper)
```

没有自定义逐 key 转换器；所有在线重命名由 `WeightsMapper` 执行。加载流程固定为：

```text
vLLM 官方 checkpoint weight iterator
-> official VibeVoice key mapping
-> PR/HF canonical module key
-> vLLM fused QKV / gate-up mapping与TP分片
-> parameter.weight_loader
```

### 7.1 Semantic Encoder

```text
model.semantic_tokenizer.encoder.downsample_layers.0.0.conv.*
  -> model.semantic_tokenizer_encoder.stem.conv.*

model.semantic_tokenizer.encoder.stages.0.*
  -> model.semantic_tokenizer_encoder.stem.stage.*

model.semantic_tokenizer.encoder.downsample_layers.N.0.conv.*
  -> model.semantic_tokenizer_encoder.conv_layers.(N-1).conv.*

model.semantic_tokenizer.encoder.stages.N.*
  -> model.semantic_tokenizer_encoder.conv_layers.(N-1).stage.*

model.semantic_tokenizer.encoder.head.conv.*
  -> model.semantic_tokenizer_encoder.head.*
```

### 7.2 Acoustic Encoder/Decoder

Encoder 采用相同的 stem/`conv_layers.N-1` 规则：

```text
model.acoustic_tokenizer.encoder.*
  -> model.audio_tower.encoder.*
```

Decoder：

```text
model.acoustic_tokenizer.decoder.upsample_layers.0.*
  -> model.audio_tower.decoder.stem.*

model.acoustic_tokenizer.decoder.upsample_layers.N.*
  -> model.audio_tower.decoder.conv_layers.(N-1).*

model.acoustic_tokenizer.decoder.stages.0.*
  -> model.audio_tower.decoder.stem.stage.*

model.acoustic_tokenizer.decoder.stages.N.*
  -> model.audio_tower.decoder.conv_layers.(N-1).stage.*

model.acoustic_tokenizer.decoder.head.conv.*
  -> model.audio_tower.decoder.head.*
```

剩余 Acoustic 前缀：

```text
model.acoustic_tokenizer.* -> model.audio_tower.*
```

### 7.3 Diffusion Head

```text
model.prediction_head.t_embedder.mlp.0.*
  -> model.diffusion_head.timestep_proj.layer_1.*
model.prediction_head.t_embedder.mlp.2.*
  -> model.diffusion_head.timestep_proj.layer_2.*
model.prediction_head.layers.N.adaLN_modulation.1.*
  -> model.diffusion_head.layers.N.linear.*
model.prediction_head.final_layer.adaLN_modulation.1.*
  -> model.diffusion_head.final_layer.linear_1.*
model.prediction_head.final_layer.linear.*
  -> model.diffusion_head.final_layer.linear_2.*
model.prediction_head.*
  -> model.diffusion_head.*
```

### 7.4 Projector 和 latent factor

```text
model.acoustic_connector.fc1.* -> model.multi_modal_projector.linear_1.*
model.acoustic_connector.norm.* -> model.multi_modal_projector.act.*
model.acoustic_connector.fc2.* -> model.multi_modal_projector.linear_2.*

model.semantic_connector.fc1.* -> model.semantic_connector.linear_1.*
model.semantic_connector.norm.* -> model.semantic_connector.act.*
model.semantic_connector.fc2.* -> model.semantic_connector.linear_2.*

model.speech_scaling_factor -> model.latent_scaling_factor
model.speech_bias_factor    -> model.latent_bias_factor
```

还要清理官方实现中的重复 Conv wrapper 层级，例如：

```text
mixer.conv.conv.conv -> mixer.conv
conv.conv.conv       -> conv.conv
```

### 7.5 Qwen2/vLLM 映射

官方 Qwen2 前缀已经是：

```text
model.language_model.*
```

在映射到实际 Omni 模型属性后，还要执行 vLLM 常规融合：

```text
q_proj/k_proj/v_proj -> qkv_proj
gate_proj/up_proj    -> gate_up_proj
```

并调用每个 parameter 的 `weight_loader` 完成 TP 分片。1.5B checkpoint 使用 tied embeddings，官方 index 不包含独立 `lm_head.weight`；加载器应将其视为 tied parameter，而不是报告 missing weight。

### 7.6 实现和验收要求

具体 canonical 映射以以下转换脚本为权威：

```text
/SharedData/youhf/transformers/src/transformers/models/vibevoice/convert_vibevoice_to_hf.py
```

不要在加载器中原地修改输入名称集合或 tensor。映射测试至少覆盖每类前缀、所有 stem/index shift、QKV/gate-up fusion，以及官方 `model.safetensors.index.json` 的全部 key。最终加载必须验证：

```text
unexpected_keys = empty
missing_keys = empty（显式允许 tied/buffer 项除外）
映射前后 tensor shape 一致
已加载 parameter 集合符合当前单 stage 模型结构
```

## 8. Processor

实现位于：

```text
vllm_omni/model_executor/models/vibevoice/processing_vibevoice.py
```

Processor 使用标准 vLLM MultiModal Processor 接口，不复用 Transformers
`VibeVoiceProcessor` 中请求共享的 `_num_audio_tokens`。职责边界为：

```text
24 kHz resample -> mono downmix -> -25 dB FS normalization
-> pad 到 3200 倍数 -> PromptReplacement
```

每个 reference audio 在 prompt 中对应一个未展开的：

```text
<|vision_start|><|vision_pad|><|vision_end|>
```

只展开 `<|vision_pad|>`，长度为 `ceil(valid_samples / 3200)`。多音频严格按
prompt occurrence、MM item、placeholder range 的顺序一一对应。Processor 显式
校验 audio item 与未展开 placeholder 数量相等；这是必要的，因为 vLLM 原生校验会
拒绝“audio 多于 placeholder”，但额外 placeholder 可能作为普通文本 token 残留。
当前基础设施容量限制为每条 60 秒（450 placeholders），每请求最多 8 条；deploy YAML
同时显式设置 `limit_mm_per_prompt.audio = 8`。最坏 reference prefill 使用
`8 * 450 = 3600` MM tokens，加上文本后仍低于 65,536 context limit；8,192 的
per-iteration budget 通过 chunked prefill 分批调度。该 8 条上限只表示 Processor/MM
预算，不表示模型能力：服务层按 Microsoft VibeVoice-1.5B 官方信封限制为最多 4 个
speaker，且要求每个 speaker 恰有一条 reference audio。

Tokenizer 解析优先级是：显式 `--tokenizer`、checkpoint 本地 tokenizer、
`preprocessor_config.json.language_model_pretrained_name`。官方当前字段值为
`Qwen/Qwen2.5-1.5B`，但实现不硬编码该名称。离线部署必须预先缓存或提供这个
tokenizer。

### 8.1 状态和风险事实源

本文档是 VibeVoice 适配的唯一事实源。Processor/Prefill 里程碑状态、upstream
placeholder 校验缺口、RNG 决策、风险表及 M3a/M3b 验收边界见第 12 节；后续不再
维护单独的 Processor plan 文档。

Processor 不进入 Config 模块，也不负责业务 prompt、speaker 映射、decode feedback
或 per-request generation state。

## 9. PR #40546 推理流程与 Voxtral 关系

### 9.1 是否以 Voxtral 为核心

PR 的 modular modeling 确实以 Transformers Voxtral 作为多模态 decoder-only 模型脚手架：

```text
VibeVoicePreTrainedModel <- VoxtralPreTrainedModel
VibeVoiceModel           <- VoxtralModel
VibeVoiceMultiModalProjector <- VoxtralMultiModalProjector
```

它复用了 Voxtral 的组织方式：音频特征经过 projector 后替换文本序列中的 audio placeholder，再送入 decoder-only language model；也复用了部分 PreTrainedModel、输出对象和多模态 forward 约定。Transformers modular 展开后，生成的 `modeling_vibevoice.py` 可能看不到显式继承，但代码来源仍是上述关系。

VibeVoice 并不是复用 Voxtral 的核心声学和生成算法：

```text
Voxtral                     VibeVoice
Whisper/Voxtral audio encoder -> continuous Acoustic Tokenizer
Llama-style text backbone      -> Qwen2
单路 audio projector           -> acoustic + semantic connectors
文本 token generation          -> token gate + diffusion latent + waveform
标准 GenerationMixin           -> 重写的 VibeVoice _sample loop
```

因此 Omni 实现可以参考 Voxtral 的多模态输入/placeholder 边界，但核心 serving 循环应参考 VibeVoice PR 和 VoxCPM2 单 stage AR 状态管理。

### 9.2 Processor 和 prefill

1. Chat template 组织 voice samples 和目标文本，并在 `Speech output:` 后追加 `audio_bos_token`。
2. Feature extractor 将参考音频归一化为 24 kHz waveform，并 pad 到 3200 samples 的倍数。
3. Processor 按 `ceil(valid_samples / 3200)` 将每个参考音频 placeholder 展开成对应数量的 `audio_token`。
4. Acoustic Encoder 将参考 waveform 压缩为连续 latents。
5. Latent 执行：

   ```text
   acoustic_features = (latents + latent_bias_factor) * latent_scaling_factor
   ```
6. `multi_modal_projector` 将 64 维 acoustic latent 投影到 Qwen2 hidden size 1536。
7. 投影结果替换 `input_ids == audio_token_id` 的 placeholder embedding。
8. Qwen2 执行正常 prefill，建立 positive branch paged KV cache。

### 9.3 AR token gate

Language model 并不直接输出音频 codec token。Logits processor 将合法输出限制为：

```text
audio_bos_token_id
audio_token_id
audio_eos_token_id
eos_token_id
```

每一步使用 argmax；PR 明确不支持基于 temperature/top-p 的 token sampling。Token 的作用是控制状态机：

```text
audio_bos -> 开始音频并重置 unconditional CFG cache
audio_token -> 触发一次 continuous latent diffusion + waveform chunk
audio_eos -> 结束当前音频段，通常随后再生成 eos
eos -> 结束请求
```

### 9.4 每个 audio token 的生成

当某个 active request 产生 `audio_token_id`：

1. 从 positive Qwen forward 取最后一个 hidden state。
2. 对 negative/unconditional branch 再执行一次 Qwen forward，维护独立 KV cache。
3. 拼接 positive/negative condition，执行 classifier-free guidance。
4. 初始化高斯 acoustic latent。
5. 使用 `DPMSolverMultistepScheduler`（默认 10 inference steps）重复调用 Diffusion Head。
6. CFG 合并：

   ```text
   guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
   ```
7. 得到 64 维 acoustic latent，并逆归一化：

   ```text
   decoder_latent = latent / latent_scaling_factor - latent_bias_factor
   ```
8. Acoustic Decoder 使用 per-request causal padding cache 解码 waveform chunk。
9. Semantic Encoder 使用自己的 causal padding cache 重新编码刚生成的 waveform。
10. 构造下一 AR step 的输入 embedding：

    ```text
    next_embedding = acoustic_projector(latent) + semantic_connector(semantic_latent)
    ```
11. 下一 Qwen decode step 使用该连续 embedding，而不是普通 `audio_token_id` embedding。

这一步“decode waveform -> semantic re-encode -> feedback embedding”解释了为什么第一版必须将 Acoustic Decoder 留在同一个 AR stage。

### 9.5 请求状态和 Omni 实现约束

每个 request 至少维护：

```text
positive paged KV cache
negative CFG paged KV cache
negative input/attention state
Acoustic Decoder padding cache
Semantic Encoder padding cache
next-step continuous embedding
noise scheduler/guidance parameters
pending/generated waveform chunks
生成状态（before_bos / generating / finished）
```

Omni 模型需要提供类似 VoxCPM2 的 per-request state 生命周期，并正确处理 batch 中只有部分 request 在当前 step 触发 diffusion 的情况。Preemption/resume 时，不能只恢复 Qwen KV cache，还必须处理上述 side-state；如果标准 scheduler 无法满足，应增加 VibeVoice 专用 AR scheduler。

最终输出直接拼接 waveform chunks，不依赖 tokenizer detokenization。

## 10. 权重加载调用链与实现设计

### 10.1 Omni/vLLM 调用链

VibeVoice 是 `LLM_AR` stage，使用 vLLM 的普通模型加载链，而不是 Omni diffusion loader：

```text
EngineCore/Worker 初始化
  -> OmniGPUWorkerBase.load_model()
  -> vLLM GPUWorker.load_model()
  -> GPUModelRunner.load_model()
  -> get_model_loader(load_config)
  -> DefaultModelLoader.load_model()
  -> initialize_model()                         # 仅构造模块
  -> DefaultModelLoader.load_weights()
  -> model.load_weights(get_all_weights(...))  # VibeVoice 模型入口
  -> process_weights_after_loading()
  -> model.eval()
```

默认 `auto`/`hf`/`safetensors` 路径最终会调用顶层模型的 `load_weights()`。`dummy`、`sharded_state`、tensorizer 或自定义 loader 不保证走完全相同的模型入口，因此不能笼统认为所有 `load_format` 都一定调用它；VibeVoice 首版只需保证默认 safetensors 路径。

Omni 自身不读取或合并 VibeVoice shard。`DefaultModelLoader.get_all_weights()` 根据 `model.safetensors.index.json` 懒迭代 `(checkpoint_name, tensor)`，顶层模型不得再次打开 safetensors，也不得把全部 1204 个 tensor 物化成 state dict。

### 10.2 `WeightsMapper` 与 `AutoWeightsLoader` 的职责

```text
WeightsMapper
  checkpoint name -> runtime module name

AutoWeightsLoader
  遍历 runtime module tree
  -> 子模块 load_weights()
  -> parameter.weight_loader()
  -> TP shard/fused parameter copy
  -> 返回 loaded parameter name set
```

`WeightsMapper` 只改名字或丢弃 tensor，不执行 tensor copy、transpose、concat。Qwen2 的 Q/K/V 和 gate/up 融合不应在 VibeVoice mapper 中重复实现：外层 mapper 保留 `model.language_model.*`，`AutoWeightsLoader` 下钻到 vLLM `Qwen2Model.load_weights()` 后，由 Qwen2 自己的 stacked mapper 和 parameter `weight_loader` 完成：

```text
q_proj/k_proj/v_proj -> qkv_proj (shard q/k/v)
gate_proj/up_proj    -> gate_up_proj (shard 0/1)
```

### 10.3 建议模块边界

Omni 当前已适配模型通常把 `hf_to_vllm_mapper` 和 `load_weights()` 放在拥有该模块的模型实现文件中，没有统一要求或普遍采用独立的 `weight_utils.py`。VibeVoice 第一版应遵循这一惯例，在顶层模型文件中定义纯名字映射 helper：

```python
def _build_vibevoice_weights_mapper(config) -> WeightsMapper: ...


class VibeVoiceForConditionalGeneration(nn.Module):
    def load_weights(self, weights): ...
```

只有当映射代码后续明显膨胀、被多个模型类共享，或者需要独立公共 API 时，再移动到 `weight_utils.py`。文件位置不是框架契约；模型拥有映射和加载语义才是契约。

mapper 应迁移 Transformers 转换脚本的 `STATE_DICT_MAPPING`，但不包含任何文件 I/O。它同时接受两种输入：

```text
Microsoft original key -> 映射到 PR/HF runtime key
已转换 HF key          -> identity，保持不变
```

主要映射域：

```text
model.semantic_tokenizer.*  -> model.semantic_tokenizer_encoder.*
model.acoustic_tokenizer.*  -> model.audio_tower.*
model.prediction_head.*      -> model.diffusion_head.*
model.acoustic_connector.*   -> model.multi_modal_projector.*
model.semantic_connector.*   -> model.semantic_connector.*
model.speech_scaling_factor  -> model.latent_scaling_factor
model.speech_bias_factor     -> model.latent_bias_factor
```

`downsample_layers.N`/`upsample_layers.N` 到 `conv_layers.(N-1)` 含索引运算，不能直接使用单个正则 replacement 表达。应根据 config 的 stage 数生成 N=1..K 的精确 regex 项，再放入 `WeightsMapper.orig_to_new_regex`；不要在热加载路径里维护 `PLACEHOLDER` 字符串协议。

### 10.4 顶层模型 `load_weights()`

建议实现：

```python
def load_weights(self, weights):
    mapper = _build_vibevoice_weights_mapper(self.config)
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=[...],
    )
    return loader.load_weights(weights, mapper=mapper)
```

VibeVoice 不需要像多 stage Omni 模型那样在顶层把 iterator 物化后手工按 component 分发。映射后的 prefix 与 runtime module tree 一致时，`AutoWeightsLoader` 会自动递归：有自定义 `load_weights()` 的子模块由子模块接管，否则继续下钻到 parameter。Qwen2 子模块会接管 fused/TP 加载，Transformers Acoustic/Semantic 模块则由通用递归加载。

预期 runtime 模块布局应与 PR key 对齐：

```text
model.audio_tower
model.language_model
model.semantic_tokenizer_encoder
model.multi_modal_projector
model.semantic_connector
model.diffusion_head
model.latent_scaling_factor
model.latent_bias_factor
lm_head  # 与 embedding tied 时无独立 checkpoint tensor
```

必须返回 runtime parameter names 的 `set[str]`。`DefaultModelLoader` 在非量化默认路径会计算：

```text
model.named_parameters() - loaded_weights
```

非空即启动失败。因此不能返回 original checkpoint names，也不应通过虚假地把所有参数标记为 loaded 来掩盖映射缺失。

### 10.5 验证要求

1. 对每条映射类别做 representative unit test。
2. 对官方 index 中全部 1204 个 key 验证：映射后无冲突、无负索引。
3. 映射后的 key set 应与 PR 转换 checkpoint index 的 1204 个 key 完全一致。
4. 对已经转换的 HF key 验证 mapper 为 identity。
5. 构造 runtime 模型后，用 synthetic tensors 或完整 checkpoint 验证：

   ```text
   unexpected source keys = empty
   missing runtime parameters = empty
   loaded set == 应加载的 named_parameters set
   ```
6. 单独验证 Qwen2 packed 参数由子模型 loader 加载，避免外层 mapper 破坏 shard metadata。
7. 覆盖 TP=1 数值核与 TP=2 完整 stateful/waveform runtime。

## 11. 测试

```text
tests/model_executor/models/vibevoice/test_vibevoice_config.py
tests/model_executor/models/vibevoice/test_vibevoice_weight_mapping.py
tests/model_executor/models/vibevoice/test_vibevoice_weight_loading_gpu.py
tests/model_executor/models/vibevoice/test_vibevoice_diffusion.py
tests/model_executor/models/vibevoice/test_vibevoice_audio_decode.py
tests/model_executor/models/vibevoice/test_vibevoice_stateful.py
tests/model_executor/models/vibevoice/test_vibevoice_processing.py
tests/model_executor/models/vibevoice/test_vibevoice_processing_gpu.py
tests/model_executor/models/vibevoice/test_vibevoice_serving_adapter.py
tests/model_executor/models/vibevoice/test_vibevoice_engine_core_gpu.py
tests/model_executor/models/vibevoice/test_vibevoice_tp2_gpu.py
tests/model_executor/models/vibevoice/test_vibevoice_async_omni_cleanup_gpu.py
tests/model_executor/models/vibevoice/test_vibevoice_full_generation_golden_gpu.py
tests/worker/test_multimodal_preprocess_contract.py
```

覆盖：

- 官方 schema 无 remote code 加载并转换；
- 转换后 HF schema 直接加载；
- 三个上游子 Config 对象化；
- Diffusion 顶层字段与 PR 消费接口；
- Qwen2 text config、dtype alias 与特殊 token；
- 原始/HF 混合子 schema 冲突检测；
- TP plan；
- round-trip；
- 输入不可变；
- vLLM resolver 与新进程注册；
- 单 stage pipeline registry 和 deploy YAML 契约；
- 全部 1204 个官方 key 与 HF 转换 index 一致性；
- DefaultModelLoader GPU 完整加载和 strict missing-parameter 检查；
- direct mapped tensor、Qwen2 packed QKV/gate-up tensor 的逐值一致性；
- 已加载 Diffusion Head 的 BF16 CUDA forward smoke test；
- model-local CFG + DPM diffusion 数值核、fresh scheduler、输入契约、Microsoft scheduler
  十步逐值一致性和官方权重 BF16 CUDA 完整 denoise loop；
- model-local inverse scaling、Acoustic Decoder causal cache、Semantic Encoder causal cache、
  acoustic/semantic feedback embedding 和官方权重双 chunk BF16 CUDA decode；
- 单 prompt 多 reference audio 的逐条 placeholder 长度与顺序；
- 60 秒/8 条/450 token 限制、超长拒绝、stereo 下混，以及 supported/user
  两层 limit 对第 9 条或超出用户配置的 audio 的拒绝；
- request-scoped UUID 的 processor hash 契约；
- `SupportsMultiModal + has_preprocess` runner 调用顺序；
- GPU Acoustic Encoder 的 `sample=False` 确定性公式对拍；
- GPU Acoustic Encoder 的 `sample=True + 固定 seed` 逐值对拍；
- per-item crop、projector、placeholder merge 和错误长度拒绝；
- 真实 `OmniGPUModelRunner._preprocess()` 驱动的 Processor→Acoustic Encoder
  →scale/bias→projector→merge GPU 组合路径；
- Serving 多说话人 prompt、audio 顺序、request/item UUID 和 model-specific sampling；
- 真实 EngineCore finish/abort、freeable、压力驱逐、hash 通知、GPU tensor 删除和
  `SamplingMetadata` 四 token mask。

CPU/映射测试：

```bash
VIBEVOICE_TEST_MODEL_ROOT=/path/to/models \
VIBEVOICE_OFFICIAL_REPO=/path/to/VibeVoice \
pytest tests/model_executor/models/vibevoice/test_vibevoice_config.py \
       tests/model_executor/models/vibevoice/test_vibevoice_weight_mapping.py \
       tests/model_executor/models/vibevoice/test_vibevoice_diffusion.py \
       tests/model_executor/models/vibevoice/test_vibevoice_audio_decode.py \
       tests/model_executor/models/vibevoice/test_vibevoice_stateful.py
```

完整 GPU 加载和 Processor prefill 测试：

```bash
CUDA_VISIBLE_DEVICES=0 VIBEVOICE_TEST_MODEL_ROOT=/path/to/models \
pytest tests/model_executor/models/vibevoice/test_vibevoice_weight_loading_gpu.py \
       tests/model_executor/models/vibevoice/test_vibevoice_processing_gpu.py \
       tests/model_executor/models/vibevoice/test_vibevoice_engine_core_gpu.py \
       tests/model_executor/models/vibevoice/test_vibevoice_tp2_gpu.py
```

## 12. Processor / Prefill 状态、决策与后续计划

本节是 Processor、多模态 prefill、M3 serving/cache 及其风险的维护入口。

### 12.1 范围和里程碑

第一版范围是 Microsoft checkpoint、非 Realtime TTS、单个 `LLM_AR` stage、
reference-audio prefill，以及最终的非流式 waveform 输出。ASR、双工/Realtime、训练
字段和独立 Code2Wav stage 不在范围内。

| Milestone                      | 状态                                                                 | 契约                                                                                                      |
| ------------------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| M0 runner composition          | 完成                                                                 | 真实`OmniGPUModelRunner._preprocess()` 在可选 model `preprocess()` 前完成 MM merge                    |
| M1 stateless Processor         | 完成                                                                 | 24 kHz、mono、-25 dB FS、3200 padding、60 秒/条、8 条/request                                             |
| M2 reference prefill           | 完成                                                                 | Registry、`SupportsMultiModal`、Acoustic Encoder、projector、per-item crop、标准 merge                  |
| M3a EngineCore/cache           | 完成                                                                 | 真实 EngineCore cache 生命周期、SamplingMetadata 四 token mask、EOS-only stop                             |
| M3b serving adapter            | 完成（prompt-level）                                                 | 请求解析、prompt 渲染、有序 MM payload、request-scoped UUID、model-specific sampling 收口                 |
| M4a diffusion numerical kernel | 完成并由 M4c 接入 runtime                                            | 显式 positive/negative condition 和 noise、fresh DPM solver、CFG、64 维 acoustic latent                   |
| M4b decode/feedback kernel     | 完成并由 M4c 接入 runtime/serving                                    | Acoustic Decoder/Semantic Encoder causal cache、3200-sample chunk、下一步 embedding、24 kHz waveform      |
| M4c stateful AR integration    | 完成（v1）                                                         | PR-0/1/2/3、真实 AsyncOmni waveform、finish/abort/exception cleanup 和 Transformers PR #40546 三步 cached full-stack golden 均通过 |

### 12.2 M1 Processor 契约和 upstream 缺口

实现位于：

```text
vllm_omni/model_executor/models/vibevoice/processing_vibevoice.py
vllm_omni/model_executor/models/vibevoice/vllm_compat.py
```

输入顺序严格为：

```text
第 N 个 <|vision_start|><|vision_pad|><|vision_end|> segment
<-> 第 N 个 audio item
<-> 第 N 个 MM kwargs item
<-> 第 N 个 placeholder range
```

只展开 `<|vision_pad|>`，长度为 `ceil(valid_mono_24khz_samples / 3200)`。预算链为：

```text
60 s * 24,000 / 3,200 = 450 MM tokens/item
8 items * 450 = 3,600 MM tokens/request
max_model_len = 65,536
max_num_batched_tokens = 8,192（chunked prefill）
```

450 是 scheduler 静态上界；`_call_hf_processor()` 负责实际时长兜底并拒绝超过 60 秒。
Deploy 显式设置 `limit_mm_per_prompt.audio = 8`。

Pinned vLLM 存在一个 placeholder 校验缺口：MM item 多于匹配 target 时会报错，但
未展开 target 多于 MM item 时，额外 `<|vision_pad|>` 可能作为普通文本 embedding
残留。VibeVoice 在 `_call_hf_processor()` 中增加双向对称数量校验，并覆盖两种回归
用例；该缺口可作为后续 upstream 贡献候选。

Stereo/multi-channel 在 `(waveform, sample_rate)` 等单 item 意图明确的路径自动下混并
`warning_once`。裸 2D ndarray/tensor 因无法区分 stereo item 与 mono batch 而直接
拒绝：单个多声道输入必须使用 tuple，多个 mono 输入必须使用 list。

私有 vLLM API 调用集中在 `vllm_compat.py`，并有签名/行为 smoke test：

```text
MultiModalDataParser._get_audio_with_sr
_merge_multimodal_embeddings
stage-0 InputProcessor.renderer.get_tokenizer
```

### 12.3 M2 embedding 契约

Runtime 顺序严格对齐 Transformers PR #40546：

```text
audio_tower.encode(sample=True)
-> sampled latent
-> (latent + latent_bias_factor) * latent_scaling_factor
-> multi_modal_projector
-> 按 item 裁剪到 ceil(mask.sum() / 3200)
-> list[Tensor(num_tokens_i, 1536)]
```

`sample=False` 仅用于确定性对拍；runtime 保持官方 `sample=True`。每个 item 必须满足：

```text
embedding[i].shape[0]
== audio_num_tokens[i]
== mm_placeholders["audio"][i].length
```

模型已注册到 Omni ModelRegistry，Processor 已通过 `MULTIMODAL_REGISTRY` 挂到
`VibeVoiceForConditionalGeneration`。GPU 已覆盖 PR 公式确定性/固定 seed 随机对拍、
真实 runner `_preprocess()`、per-item crop、merge 和 8 条 × 60 秒 H100 BF16 eager
上界。独立的 M3a GPU 测试现已覆盖真实 EngineCore scheduler/cache plumbing。

### 12.4 Tokenizer、stop 和 valid-token 状态

Tokenizer 解析优先级：

```text
显式 --tokenizer
-> checkpoint 本地 tokenizer
-> preprocessor_config.json.language_model_pretrained_name
```

官方 metadata 当前指向 `Qwen/Qwen2.5-1.5B`，代码不硬编码；离线部署必须预缓存或
传本地 tokenizer 路径。

PR 控制 token 契约已验证：

```text
allowed_token_ids = [
  151652,  # audio_bos
  151653,  # audio_eos
  151654,  # audio diffusion placeholder
  151643,  # eos
]
stop_token_ids = [151643]
```

`audio_eos=151653` 是 AR 相位转换而不是 stop。四 token 集合、EOS-only stop 和
`detokenize=False` 只定义在 pipeline `sampling_constraints`；deploy YAML 不重复这些
正确性不变量。Pinned vLLM 的 `SamplingParams.allowed_token_ids` 和 stock v1 Sampler
已经过单测，能够把其余 vocabulary logits 置为 `-inf`，无需自定义 VibeVoice logits
processor。

已确认请求级缺口：唯一 choke point 是
`vllm_omni/entrypoints/omni_base.py::resolve_sampling_params_list()`。调用者传入完整
`sampling_params_list` 时列表原样直通，最终可使 `allowed_token_ids=None`，从而绕过
pipeline constraints；bare `sampling_params` 路径也汇入这里。另一个前置问题是
`_build_extras()` 已把 constraints 合并进 `default_sampling_params`，请求层无法再区分
普通 deploy default 与不可覆盖的 pipeline key。

曾评估把独立 constraints 贯通 stage config/client/engine，并在 OmniBase 统一执行
clone + per-key replace；该通用 runtime 修改已撤回，避免影响所有现有模型。M3 已在
`VibeVoiceTTSAdapter.apply_sampling_overrides()` 完成 model-specific 收口：正常 TTS
请求基于 stage defaults 构造参数，并在提交前重新应用四 token、EOS-only stop 和
`detokenize=False`。Serving 调用处同时检查 `_tts_model_type == "vibevoice"` 和
`adapter.name == "vibevoice"`；未向所有 TTS adapter 开放通用 finalize/sampling hook。
对象和 dict 均使用 per-key replace，默认路径幂等，且不修改调用者参数。低层调用者
完整替换 `sampling_params_list` 的绕过能力仍是已知高级 API 限制，不为此修改共享
Omni runtime。

Waveform E2E 已跑通，因此 adapter 现在会在用户 `extra_params` 之后 model-specific 地
恢复 `temperature=0.0`，与官方非 Realtime generation 写死的 argmax 一致。该温度只
作用于 Qwen2 hidden state 经 tied LM head 后、四个合法控制 token 之间的选择，不作用于
Diffusion Head、DPM solver 或 acoustic latent。对象和 dict 路径都已验证 clone、幂等和
调用者不变；未修改共享 sampler，也保留 deploy 中正确的默认值。低层完整
`sampling_params_list` 绕过 adapter 仍属于上文已登记的高级 API 限制。

Pinned vLLM 的 text-prompt preprocessing 路径不会把 `multi_modal_uuids` 传给 MM
Processor。Adapter 因此同时提交 `prompt` 和 model-specific tokenizer 生成的
`prompt_token_ids`，强制走保留 UUID 的 token-prompt 路径；未修改 vLLM/Omni runtime。
`audio_eos -> eos` 的实际 decode 相位转换留到 M4。

### 12.5 RNG 决策

默认保持官方全局设备 RNG。UUID 解决跨请求 encoder-cache 共享，但不隔离随机流；
batch 组成和调度顺序可能改变单请求 sampled latent，这是与官方一致的默认语义，不
阻塞 M3。

如未来要求 batch-order-independent reproducibility，可选方案为：

```text
seed = stable_hash(request-scoped UUID，其中包含 item index)
per-item torch.Generator
```

该模式登记为 M4 决策，不得静默替换默认官方行为。

### 12.6 M3a：真实 EngineCore/cache（完成）

`test_vibevoice_engine_core_gpu.py` 不依赖 Serving 或 waveform decode，使用官方权重和
真实 EngineCore plumbing，已完成以下验收：

1. 相同 waveform + 相同 UUID 只执行一次 encoder；
2. 相同 waveform + 不同 UUID 重新执行 encoder；
3. finish 和 abort 移除 request reference，使零引用 entry 进入 `freeable`；
4. 后续容量压力从 `freeable` 驱逐，`free_encoder_mm_hashes` 发出 hash，GPU runner
   删除对应 tensor；
5. 在 VibeVoice model-specific serving/sampling 路径固定重新应用四 token、EOS-only
   stop 和 `detokenize=False`，不修改共享 Omni runtime；
6. 负向用例覆盖对象和 dict：调用者显式参数不含或传错约束时，正常 VibeVoice TTS
   请求最终仍使用 pipeline 值；默认值路径重复应用必须幂等；
7. 验证默认 pipeline/正常 serving 路径的四 token mask 真实到达 EngineCore
   `SamplingMetadata`；低层完整 `sampling_params_list` 绕过作为已知限制登记。

Pinned v1 runner 的通用 MM profiler 会先以“有 dummy placeholder、无 MM data”的文本
路径调用 Processor，从而正确触发 VibeVoice 的对称 placeholder 校验。Deploy 采用固定
KV cache bytes，且已有独立 8×60 秒真实 encoder 上界测试，因此设置
`skip_mm_profiling=true`，避免不兼容的通用 profiler 调用而不放松请求校验。

Encoder cache 是两阶段生命周期：finish/abort 只把 entry 变为可回收；只有后续
`can_allocate()` 遇到容量压力才物理驱逐并写入 `freed`，随后
`get_freed_mm_hashes()` 返回并清空通知。测试不能期待请求结束立即删除 GPU tensor。

Request-scoped UUID 会同时阻止前端 Processor cache 和 GPU encoder cache 的跨请求
命中，这是预期行为；Processor cache 只保留同 UUID 范围内的复用，不专门调大
`mm_processor_cache_gb`。

### 12.7 M3b：Serving adapter（prompt-level 完成）

实现位于 `entrypoints/openai/tts_adapters/vibevoice.py`。M3b 不依赖 M4 decode，已验收
请求转换：

```text
request -> (prompt, multi_modal_data, multi_modal_uuids)
```

Adapter 负责：

- 请求解析和目标文本；
- 多说话人 prompt 渲染；
- speaker/audio 精确顺序；
- 每条 reference audio 一个未展开 vision segment；
- `(waveform, sample_rate)` item 表示；
- `{request_id}:audio:{item_idx}` UUID；
- 不依赖非空 `hf_processor_mm_kwargs`。

服务层按 VibeVoice-1.5B 官方能力限制最多 4 个 speaker；Processor 和 deploy 的
`MAX_AUDIO_ITEMS=8` 只保留为 MM 基础设施容量。`response_format` 由共享 audio serializer
处理，非流式 `speed` 由共享后处理真实生效。单 speaker 请求可使用已通过
`POST /v1/audio/voices` 上传的音频 `voice`，语义等价于直接提供该音频的 `ref_audio`；
`voice` 与显式 `ref_audio` 互斥，多 speaker 仍要求按首次出现顺序直接提供 reference list，
embedding-only uploaded voice 不支持。Adapter 的 validate 保持只读和幂等，build 才把
uploaded voice 解析为 canonical `ref_audio`，并清除通用 registry 可能附带但 VibeVoice
不消费的 stored `ref_text`。

VibeVoice 未消费的通用 Speech 字段（`speaker_embedding`、`instructions`、`language`、
`ref_text`、`ref_audio_2`、`task_type`、`ambient_sound`、`duration_seconds`、
`x_vector_only_mode`、`initial_codec_chunk_frames`、`non_streaming_mode`、
`word_timestamps`）在显式设置时返回 400，不允许静默忽略；`extra_params` 作为模型扩展
通道继续允许未知键前向兼容。Batch Speech 第一版只承诺单 speaker item（协议中的
per-item `ref_audio` 是单字符串）；每个 item 是独立 single pass，整体 HTTP 200 内以
`results[i].status=success|error` 隔离错误，成功项以 `finish_reason=stop|length` 暴露终止
原因。PCM DELTA 拼接以逐比特相同为验收口径；streaming WAV 因 header 长度字段不同，只
比较解码后的样本，不比较完整文件 bytes。

低层 `LLM.generate()` 是高级 API，调用者必须显式提供 request-scoped UUID；Processor
不得隐式生成随机 UUID，否则会破坏同请求 chunked-prefill 复用。完整 waveform E2E
已在 M4 完成。

### 12.8 M4 门槛和 golden reference

TP=2 gate 已完成：vLLM Qwen2 的 packed QKV/gate-up 按 rank 分片，Acoustic/Semantic
Encoder、projector、Diffusion Head 和 latent scale/bias 在各 rank 复制；相同输入的
projector/diffusion 输出逐值一致。覆盖位于 `test_vibevoice_tp2_gpu.py`。

Serving/API 契约还包括：HTTP SSE（`stream=true` 或 `stream_format=sse`）和 WebSocket
`stream_audio=true` 使用现有 DELTA waveform 通路；SSE 的 `speech.audio.done` 与 WebSocket
的 `audio.done` 都携带 `finish_reason=stop|length`。无法在响应结束时携带结构化终止状态的
raw PCM/WAV HTTP streaming（`stream_format=audio`）继续在提交 Engine 前拒绝，避免重新
引入静默截断；WebSocket streaming 按共享协议只接受 PCM。显式 request `seed` 返回 400，
deploy 默认 seed 不改变官方 waveform RNG；显式 `guidance_scale` 必须为有限数值，
`num_diffusion_steps` 必须为非 bool 正整数。Adapter 与 stateful runtime 共用同一校验函数，
runtime 校验仍作为低层防御。非流式 HTTP 音频响应保持 200，并以
`X-Finish-Reason: stop|length` 暴露最终终止原因；`length` 明确表示 token cap 截断，不能
解释为内容已完整生成。4-speaker CPU contract 已验证，但此前声称的两次“真实 HTTP”
运行实际由 pytest 默认 `--run-level=core_model` 注入了 `load_format: dummy`；随机权重在
512/1024 token 达到 length cap 对真实 checkpoint 没有信息量，相关 bug 定性、动态 xfail
和 activation 结论均已撤回。4-speaker HTTP E2E 现在要求 advanced/full run level、在启动
前拒绝 dummy stage config，并严格要求 `finish_reason=stop`；随后真实 checkpoint TP=2
advanced-model 运行已通过。test-only lifecycle trace 使用相同四条 reference 时也自然停止、
chunk 数与 audio-token 数一致且双 rank negative branch 全释放；该次模型生成了 3 个平衡的
BOS/EOS audio segment。输出 segment 是模型自行选择的连续音频边界，公开 runtime 不规定其
必须与 4 个输入 speaker turn 一一对应，因此不能把 `audio_eos` 数必须为 4 当 correctness
契约；是否完整朗读四个 turn、speaker consistency 如何仍需 ASR/alignment/diarization 质量
评测。约 90 分钟探针同样只验证容量、终止和运行稳定性，不代表长程内容准确率。

#### 12.8.1 F6 质量评测状态

第一阶段非阻断评测已接入现有 Seed-TTS judge，而不是复制一套 ASR/SIM 实现：

- `seed-tts-vibevoice` dataset variant 只发送 VibeVoice 支持的 `ref_audio` 和
  `max_new_tokens`，不发送 Qwen 专用的 `ref_text`、`task_type`、`language`；
- benchmark client 使用 OpenAI Speech SSE 捕获 PCM 和 `stop|length` terminal metadata，
  保持 raw HTTP streaming 的拒绝契约；VibeVoice dataset 显式要求 terminal `stop`，non-stop
  音频只保存为诊断 artifact，不执行 ASR/SIM/UTMOS，也不纳入 CER/WER aggregate；
- `tests/e2e/accuracy/vibevoice/run_vibevoice_quality.py` 支持英文 WER、中文字符级 error
  rate、WavLM reference similarity、per-item JSON 和 WAV artifact；
- 阈值默认关闭，必须先取得 Microsoft/Transformers PR 与 Omni baseline 后才能升级为
  nightly blocking gate。

已有服务运行时可执行：

```bash
python tests/e2e/accuracy/vibevoice/run_vibevoice_quality.py \
  --model VibeVoice \
  --dataset-path /path/to/seed-tts-eval \
  --locale both \
  --num-prompts 8 \
  --max-concurrency 2 \
  --save-audio-dir /tmp/vibevoice-quality-audio
```

当前 F6 仍未完成：judge checkpoint revision 尚未在 VibeVoice gate 中独立锁定，阈值尚无
baseline，Seed-TTS 第一阶段只覆盖单 speaker；2/4-speaker turn-level similarity 仍需要固定
多说话人 corpus 和离线 alignment/diarization 方案。不得把第一阶段结果描述为完整 F6 通过。

Golden oracle 边界必须明确：Microsoft 官方 Git 历史没有公开发布与 1.5B checkpoint
匹配的完整非 Realtime `generate()`（usage 因滥用风险关闭；当前仓库中的
`modeling_vibevoice_streaming_inference.py` 属于 Realtime/Streaming 架构，不能作为本模型
oracle）。因此 Microsoft 原始 Processor、Acoustic/Semantic tokenizer、scheduler 和公式是
component-level oracle；Transformers PR #40546 是当前公开的 full-generation oracle。
Omni 必须同时满足两层对拍，不能只与 PR 自洽。

M4 v1 correctness 已完成；合入前仍需独立部署验收项：

- 明确最低 GPU 显存支持，或增加小卡完整 engine profiling 测试；
- final-step per-request state 与 finish/abort/exception cleanup 已由真实 TP=2 AsyncOmni 验证；
- Transformers PR #40546 三步 cached full-stack generation golden 已自动化。M2 已证明 PR
  prefill 公式等价；test-only helper 从 Microsoft 官方三 shards 在线映射权重，因此缺失的
  converted HF shard 不阻塞关键路径。

#### M4a Diffusion 数值核（完成）

实现位于 `model_executor/models/vibevoice/diffusion.py`：有权重的
`VibeVoiceDiffusionHead` 负责单 timestep prediction，无参数的
`VibeVoiceDiffusionSampler` 负责 CFG 和多步 DPM 求解；`VibeVoiceModel` 只负责实例化和
调用。该模块不修改 Omni runtime，不拥有 request、Qwen KV、decoder cache 或共享
scheduler state。每个 audio-token 调用显式接收：

```text
positive_condition [B, 1536]
negative_condition [B, 1536]
noise              [2B, 64]
guidance_scale
num_inference_steps
```

每次调用创建 fresh `diffusers.DPMSolverMultistepScheduler`，执行官方 CFG 组合并返回
`[B, 1, 64]` acoustic latent。`cosine` 在模型 view 中规范化为
`squaredcos_cap_v2`；固定输入下，当前 diffusers 0.38.0 与 Microsoft
`vibevoice/schedule/dpm_solver.py` 的 betas、timesteps 和十个 `prev_sample` 逐值完全一致。
完整官方权重的 BF16 CUDA 十步 denoise loop 已通过。

M4a 故意要求调用者提供完整 noise，不在数值核中决定 CPU/GPU RNG、seed 或 batch-order
语义；noise 创建和 per-request RNG 归 M4c。另一个已登记的配置差异是：Microsoft 原始
model config 的 `ddpm_num_inference_steps=20`，而转换后 HF `generation_config.json` 和
deploy correctness default 是 10。数值核允许显式覆盖，M4c 必须从最终 request/deploy
生成参数传入 10，不能依赖原始 model-config fallback。当前两个 scheduler 在 NumPy 2
环境都会从内部 `np.array(torch_tensor)` 发出 deprecation warning，不影响数值结果，
属于上游 diffusers/Microsoft scheduler 兼容性问题。

#### M4b Decode/feedback 数值核（完成）

实现位于 `model_executor/models/vibevoice/audio_decode.py`。无参数的
`VibeVoiceAudioTokenDecoder` 接收 `[B,1,64]` acoustic latent 和调用者拥有的两套 cache，
严格执行：

```text
decoder_latent = latent / latent_scaling_factor - latent_bias_factor
-> Acoustic Decoder(use_cache=True) -> [B,1,3200] waveform
-> Semantic Encoder(use_cache=True)  -> [B,1,128] semantic latent
next_embedding = acoustic_projector(original latent)
               + semantic_connector(semantic latent)
-> [B,1,1536]
```

kernel 返回 waveform、semantic latent、next embedding 和更新后的 acoustic/semantic
cache，但不保存 request state。CPU fake-module contract 已验证 inverse scaling、原始 latent
feedback、两套 cache 独立传递、输入不原地修改和 shape/error surface。官方完整权重 BF16
CUDA 已连续解码两个 chunk；每个 chunk 为 3200 samples，feedback 为 `[1,1,1536]`，全部
finite。cached Acoustic Decoder 两个 chunk 拼接与一次性 6400-sample causal decode 通过
默认 BF16 close。

测试发现 Semantic Encoder 的 BF16 cached chunk 路径与一次性 full-sequence 路径不
bit-exact：H100 上固定输入最大绝对差为 0.125（首 chunk 也有 0.0664），原因是不同输入
shape 选择的 BF16 convolution 数值路径不同。官方 generation 使用逐 chunk cache，因此
后续 Transformers PR full-stack golden 必须比较相同 cached execution，不能把 full-sequence bit equality
当作 correctness 条件；当前测试保留 `max_abs_diff <= 0.25` 的有界回归保护。

另一项 M4c 风险是 Transformers padding cache 内部 tensor 以 batch index 存储，而 vLLM
batch 会动态重排且每步只有部分 request 产生 `audio_token`。M4b 只定义 caller-owned
cache 输入/输出；M4c 必须使用 per-request cache（第一版可逐请求执行），不得共享一个
全局 batched cache。后续有真实性能数据后再评估 pack/unpack batching。

#### M4c phase 1 request-local state/hooks + PR-1/PR-2 negative branch（完成）

实现位于 `model_executor/models/vibevoice/stateful.py`。模型已启用 Omni 现有
`has_preprocess` / `has_postprocess` / `on_requests_finished` hooks，不修改 shared runtime：

- `preprocess()` 以 runner 注入的 `request_id` 建立 request state；chunked prefill 只在
  `num_computed_tokens == 0` 重置，最终 prompt token 为 audio BOS 时初始化 audio segment；
- `postprocess()` 只保留当前 request 最后一个 positive Qwen hidden row，且声明
  `requires_full_prefix_cached_hidden_states=False`；
- 每次 positive Qwen forward 前按 request 保存它实际消费的最后一行 embedding；已绑定的
  negative owner 在下一个 `audio_token` transition 时消费该 embedding，发布对齐 hidden；
- `audio_token` transition 延后到同一 runner forward 前执行；相同 guidance/steps 的 active
  subset 使用一次 global-device RNG `[2B,64]` 和 batched M4a，M4b 再按 request cache
  逐项 decode 并写回对应 flattened input row；
- `audio_eos_token_id` 只结束 audio segment，只有 model EOS 才由 scheduler 停止；
- condition 是严格 one-step value，M4a/M4b 后立即清空，防止负分支错位时复用旧 hidden；
- waveform chunk 先暂存为 request-local CPU FP32，`make_omni_output()` 每步以 sparse
  request routing 转移所有权并清空 model-local 列表；OutputProcessor 按请求沿最后一维
  累积，发布 mono 24 kHz waveform 和 `sr=24000`；
- finish/abort cleanup 采用 VoxCPM2 相同的 deferred 模式，因为 runner 在当前 forward 前
  调用 `on_requests_finished()`。下一个 request preprocess 清理无 final forward 的 abort；
  finished-and-scheduled request 在自己的 postprocess 后清理。

当前正常 AsyncOmni 路径会自动绑定 negative branch；若 runner capability 缺失或分支错位，
进入 `audio_token` 时仍会明确抛出：

```text
VibeVoice audio_token requires an independent negative Qwen PagedAttention branch.
```

禁止静默退化为 guidance=1 或复用 positive hidden。另一个已确认边界是：直接
`vllm.LLM` 使用上游 `vllm.v1.worker.GPUModelRunner`，不会安装 Omni 的 named-KV
capability；该路径继续只用于 Processor/Encoder-cache/prefill 回归，完整 M4c waveform
必须经 `AsyncOmni -> GPUARModelRunner`。若要支持低层 `vllm.LLM`，必须修改上游 runner
接触面，属于当前计划外 runtime 工作，已记录但不实施。CPU contract 已覆盖双 chunk cache、
control override、active-subset global RNG `[2B,64]`、condition 单次消费和 deferred
cleanup；真实 `OmniARAsyncScheduler -> GPUARModelRunner` 已强制生成两个 audio token，
完成一次 `negative Qwen -> M4a -> M4b -> positive feedback` transition；官方 checkpoint
负分支连续两步 hidden finite，TP=2 每 rank 独立负 KV 且聚合 hidden 一致。低层 EngineCore
Processor/Encoder-cache 生命周期继续通过。

#### M4 CFG/Paged KV 现有方案调查与决策点

当前仓库有三类近似方案：

1. **Audex CFG companion（真正的 AR PagedAttention 双分支）**：parent/`uncond` 是同一
   Omni AR engine 中的两个完整 request，各自由 scheduler/KVCacheManager 拥有原生 paged
   KV；`prompt_expand_func` + `CfgCompanionTracker` 负责创建、输出抑制、abort/cleanup，
   `AudexCFGLogitsProcessor` 的 pair-aware patch 保证两个 request 同步前进并复制 token。
   这适合 cond/uncond prompt 等长且每步 token-identical 的 logits CFG。
2. **experimental AR-Diffusion named KV branches**：
   `ARDiffusionModelRunner` 拥有 `positive`/`negative` 独立 `KVCacheManager` adapter、block pool、
   session reset/close，并通过 capability context 临时绑定给模型。这是 ownership 上最接近
   VibeVoice 的设计，但当前只服务 DiffusionEngine、单 request、非 causal attention，不能
   直接接 Omni LLM_AR/Qwen2。
3. **Qwen3-TTS/Fish Speech nested predictor**：短时子 AR 在 model 内使用独立 vLLM config
   或预分配 contiguous KV，生命周期限于单次 codebook loop；不适合 VibeVoice 最长 40.5k
   step、可抢占/恢复的 negative Qwen branch。

Audex companion 不能直接用于 VibeVoice，原因不仅是当前
`AsyncOmniEngine._enqueue_cfg_companions()` 仅在 `final_stage_id > 0` 启用。官方 negative
branch 从一个 audio BOS 开始，长度与 positive prompt 不同，并在每个新 audio BOS 把
context 重置为单 token；之后只在 `audio_token` 上消费与 positive 相同的 continuous
feedback embedding。Audex 的“相同绝对 progress + 相同 sampled token”同步假设不成立，
简单 padding 会被 causal attention 看见，也不等价。

在“不修改 runtime”约束下，现有 Omni LLM_AR 没有可正确表达该 shorter/resettable branch
的 public hook；不能用 Audex companion 或 model-global contiguous KV 伪装完成。设计评审
已完成，结论：增加一个通用、model-neutral 的 runner-owned named causal KV branch
capability（参考 AR-Diffusion ownership，但复用 Qwen2 causal PagedAttention）。决策锁定
前 M4c phase 1 保持 fail-fast，shared runtime 不变。

#### M4c negative KV branch 架构与分阶段执行计划

本小节是该能力的唯一事实源。vLLM 本体零改动；共享 runner 仅纯增量且默认关闭
（模型不声明能力则零行为变化）。

修订记录：v4 保留 runner-owned NamedCausalKVBranch 最终 ownership，并在完成单请求
Microsoft waveform E2E 后落地固定安全并发：默认 `tensor_parallel_size=2`、
`max_num_seqs=2`、每 rank 4 GiB negative GPU pool、无 swap、无自定义 scheduler。
正负两池均以 `max_num_seqs * max_model_len` 做启动守卫；negative Qwen 按请求顺序进入
独立 attention context，同 controls 的 active subset 用一次 `[2B,64]` 官方 RNG draw。
CPU arena、swap 和动态 reservation scheduler 仍不实施，只有 profiling 证明固定容量
无法满足目标负载时才立项。D1 继续使用
`VllmConfig.additional_config`；bind 期按实际 page/block/TP 布局做容量和显存 pre-flight。
最终 ownership：

```text
positive Qwen KV      -> 现有 GPUARModelRunner / Omni scheduler
acoustic/semantic     -> VibeVoice per-request state（已实现）
negative Qwen KV      -> runner-owned NamedCausalKVBranch（本小节）
negative hidden       -> `VibeVoiceNegativeKVBranch.forward_step()` 仅发布 one-step condition
```

四项决策：

**D1 配置通道（模型私有，框架零感知）**：负池预算走 `engine_extras.additional_config`
（`EngineArgs.additional_config` -> `VllmConfig.additional_config` 通用通道），不动框架
schema，也不写入 HF Config（遵守“Config 层不承载部署参数”边界，且无需为 @strict 的
`VibeVoiceConfig` 增加声明字段）：

```text
deploy/vibevoice.yaml:
  stages[0].engine_extras.additional_config.vibevoice_runtime_config:
    negative_kv_cache_memory_bytes: 4294967296       # 4 GiB，缺省同值
    negative_kv_activation_margin_bytes: 536870912   # 512 MiB，缺省同值
```

解析侧新增 `models/vibevoice/runtime_config.py`（类型化 dataclass，仿
`voxcpm2/runtime_config.py`：默认值、coerce、未知键 warning、下限校验），读取
`vllm_config.additional_config["vibevoice_runtime_config"]`。容量公式（TP=1）：
28 层 x 2(K,V) x 2 kv heads x 128 x 2 B = 28,672 B/token；4 GiB 在 TP=1 估算约
149.8k negative tokens。实际默认 TP=2 启动日志按分片 page 布局得到 299,584 tokens；
实现始终按实际 page/block 布局校验 `2 * max_model_len`，不使用估算值，也不分配 swap
arena。deploy
契约对用户不感知：框架内置
yaml 经 `default_deploy_config_name` 按 model_type 自动命中，高级用户可自带 deploy
覆盖。

**D2 构造位置**：runner 只构造 model-neutral 的 store，不 import 任何模型代码；模型在
`bind_named_kv_branch(store)` 内自行包装 executor。import 方向保持模型 ->
`vllm_omni.worker`（先例：`voxcpm2_talker.py` import
`vllm_omni.worker.runner_assisted_metadata`）。

**D3 v1 容量语义（固定容量、禁止超卖）**：首版不实现 CPU swap，也不允许负池运行时
耗尽。容量在启动时按实际 `kv_cache_spec.page_size_bytes`、block size、TP 后 KV heads 和
每 rank tensor 布局校验，文档中的 28,672 B/token 只作为 TP=1 估算，不作为实现常量：

```text
C1 固定并发负分支可容：
     negative_pool_blocks >= max_num_seqs * ceil(max_model_len / block_size)
     max_model_len=65,536 是低层请求绝对上界，不能只依赖 deploy 默认
     max_tokens=40,500
C2 固定并发正分支可容：
     positive_pool_tokens >= max_num_seqs * max_model_len
C3 默认并发固定：
     tensor_parallel_size=2（side modules 每 rank replicated）
     max_num_seqs=2
     正负两池都完整容纳两个最大请求，KV 压力抢占构造性不可达；真实默认启动日志为
     positive=449,376 tokens/rank、negative=299,584 tokens/rank，均大于 131,072
```

任一守卫不满足都在启动期明确失败；运行时不以 forward 异常表达普通容量不足。由于
两个请求的正负完整 working set 都可常驻 GPU，当前实现不需要 victim、arena、swap 或
embedding replay，也不改变官方 global-device RNG 语义。测试 worker 的 concurrency trace
现同时记录每 rank start/end/peak allocated、reserved 和 free bytes；下一次干净 TP=2 默认
部署回归将把真实双请求 activation peak 固化到验收记录中，该诊断不进入生产 runtime。

**容量扩展顺序**：waveform E2E 和 TP=2 stateful 通过后，先增加 negative GPU pool，并将
固定并发提高到实际安全值：

```text
safe_concurrency = min(
    floor(positive_pool_blocks / positive_blocks_per_max_request),
    floor(negative_pool_blocks / negative_blocks_per_max_request),
)
max_num_seqs <= safe_concurrency
```

若不限制低层 `max_tokens`，必须按 `max_model_len` 计算；只有 serving/input 层形成不可绕过
的请求上限时，才能用更小的 per-request cap。固定安全并发仍使用现有
`OmniARAsyncScheduler`，不需要模型私有 scheduler。

**可选超卖设计（非 v1）**：只有 profiling 证明固定 GPU 容量不能满足目标并发时，才实施
active-subset 保序 microbatch、CPU arena、块级 swap 和动态 reservation scheduler。届时
必须满足：GPU pool 可容单个最大负序列；GPU+预分配并 touch 的 host arena 可容所有已
准入请求的逻辑上界；microbatch 只保护当前执行批，后续 microbatch request 允许先换出；
全部 negative conditions 收齐后再统一抽取 M4a `[2B,64]` noise。持续活跃请求导致的 PCIe
抖动必须有明确 profiling 门槛和背压策略，不能仅以“只慢不错”视为可部署。

**D4 Protocol 收敛**：`VibeVoiceNegativeKVBranch` 收敛为 `reset_audio_segment` /
`forward_step` / `free` 三方法；`on_requests_finished` 从直接调用路径撤下。runner 在
当前 forward 之前调用 `on_requests_finished`（gpu_ar_model_runner.py），finished 请求可能
仍有最后一次调度；free 必须挂在 stateful 的 deferred 清理汇聚点 `cleanup_request`
（`flush_deferred_cleanup` / `finish_postprocess` 两条路径汇聚），模型 bind 时把 branch
引用交给 stateful。另注：abort 后若再无新请求（真 idle），deferred cleanup 会延迟到
下一次具有完整 scheduled request 集合的 model forward 或进程退出才释放——占用有界
（<= max_num_seqs 份），不影响正确性。不能在逐请求 preprocess 或 step 末无条件 flush：
前者尚不知道同 batch 后续请求，后者在 async scheduling 下无法判断 finished request
是否已进入下一步。当前在 model forward 开始时用完整 scheduled set 作为 exclude，只清理
不在该集合中的 deferred request；final postprocess 继续承担另一条安全清理路径。

**机制事实**（pinned vLLM 源码核实，方案的事实基础）：

1. `get_attention_context`（model_executor/layers/attention/attention.py）按 layer_name 取
   四元组：`forward_context.attn_metadata[name]`、`no_compile_layers[name]`（即
   `compilation_config.static_forward_context`）、`attn_layer.kv_cache`（**layer 属性**，
   `bind_kv_cache` 一次性绑定）、`forward_context.slot_mapping[name]`。四者中仅
   kv_cache 不来自 context——"换 store = 临时换属性 + 嵌套 context"由此成立。
2. `override_forward_context`（forward_context.py）保存/恢复全局 context，可嵌套；
   vLLM 自身在 ubatch capture 中使用同一机制。
3. `Qwen2Model.forward(input_ids=None, positions, inputs_embeds)` 在 PP=1 时可独立调用，
   除 attention 层读 context 外无 runner 依赖。
4. `FlashAttentionMetadataBuilder(kv_cache_spec, layer_names, vllm_config, device)` 的
   `build(common_prefix_len, CommonAttentionMetadata)` 只消费 query_start_loc / seq_lens /
   block_table_tensor / slot_mapping / num_reqs / max_query_len / max_seq_len / causal。
   negative batch 是退化形态（每请求 1 query token、无 prefix cache / cascade / spec
   decode），可手工构造。必须使用**独立 builder 实例**（不与 runner 共享内部缓冲区）。
5. 负池 tensor 以 runner `self.kv_caches[i]` 为形状模板（只改 block 维度）——布局随
   backend/版本变化（当前 FA 为 K/V 打包在末维），天然继承 TP 分片；各 rank 决策
   确定性一致，无需 TP 专属代码。

**显存核算（独立校验，v1 不做预算合并）**：deploy 固定 `kv_cache_memory_bytes` 后
vLLM 跳过自动 KV sizing，负池不在其预算内。bind 发生在权重 + 正池分配之后，
`torch.cuda.mem_get_info().free` 即真实剩余，bind 时校验：

```text
require: free_vram >= negative_pool_bytes + activation_margin（文档给估算式）
fail:    启动报错，含 权重/正池/负池/余量 四件套公式
```

演进项：`kv_cache_memory_bytes` 未固定时在 determine_available_memory 阶段先扣负池
预算再算正池（预算合并核算），待第二个 capability 用户出现时再做。

**v1 组件与文件**：

```text
vllm_omni/worker/named_kv_branch.py（新增，固定 GPU pool）
  NamedKVBranchRequest（frozen dataclass：name/memory_bytes/layer_group/activation_margin）
  NamedCausalKVBranch：负池 tensor + free-list allocator + 独立 builder
    + reset/free/append_and_enter（换属性 + 嵌套 context 的上下文管理器）
    + bind 期实际容量/显存 pre-flight + 固定 `max_num_seqs` 个独立 request state
vllm_omni/worker/gpu_model_runner.py（纯增量、默认关闭）
  initialize_kv_cache 覆写：super() 后 _maybe_bind_named_kv_branches()
  （getattr(model, "named_kv_branch_request", None) 为 None -> 零行为变化；
  is_profiling 跳过；逐项守卫校验）
models/vibevoice/negative_branch.py（新增）
  VibeVoiceNegativeBranch：多请求按序进入独立 KV context 的 Protocol 实现
models/vibevoice/runtime_config.py（新增，D1，读 vllm_config.additional_config）
models/vibevoice/vibevoice.py
  named_kv_branch_request 属性、bind_named_kv_branch、删 on_requests_finished 转发
models/vibevoice/stateful.py
  Protocol 收敛、bind_negative_branch、cleanup_request 调 branch.free
models/vibevoice/pipeline.py / deploy/vibevoice.yaml
  继续使用 OmniARAsyncScheduler；默认 max_num_seqs=2 和 4 GiB negative pool
```

非 v1 组件包括 CPU arena、swap/victim、active-subset microbatch 和
`VibeVoiceOmniARAsyncScheduler`；不得为提前保留接口而进入 correctness PR。

**执行时序**（一次 audio-token step 的 model.forward 内）：

```text
negative forward_step: append 1 slot/请求 -> 手工 CommonAttentionMetadata
  -> 独立 builder.build -> 交换 28 层 kv_cache -> override_forward_context
  -> Qwen2Model(inputs_embeds=负序列末条 embedding) -> 取末行 hidden
  -> finally 恢复 28 层属性（context 自动恢复）
-> record_negative_condition -> M4a CFG -> M4b 逐请求 decode
-> 反馈 embedding index_copy_ 写回 -> positive forward
```

negative 序列 = [BOS_embedding, E0, E1, ...]，RoPE 位置 = 序列内下标；比 positive 晚一个
embedding 消费（同一 E 流，wall-clock 差一步），两侧 hidden 按"消费同一 embedding"配对。

**正池抢占防护（v1）**：v1 RECOMPUTE 抢占对 VibeVoice 不可恢复（E 历史未保留、
conv cache 不能由 token IDs 重建、重放 audio 位会退化为普通 token embedding）。首版不以
自定义 scheduler 修补该问题，而是通过 C1/C2 的 `max_num_seqs * max_model_len` 启动守卫
保证所有固定准入请求的完整正负序列可常驻，使 KV 压力抢占构造性不可达。仍使用现有
`OmniARAsyncScheduler -> GPUARModelRunner`；任意 deploy 覆盖值都必须同时满足正负容量
守卫，否则启动失败而不是运行时 warning 或 preemption。

**并发扩展**：优先把正负 GPU pool 同时扩大到固定安全并发，并继续使用现有 scheduler。
只有需要允许 `max_num_seqs` 大于 worst-case safe concurrency 时，才设计模型私有动态
reservation scheduler；届时必须按每个 admitted request 的完整 block 上界记账，不能只算
剩余增长，也不能仅覆写 `_should_defer_waiting_admission()`。请求级拒绝必须落在具有
request-local error handling 的 input/preprocess 层，不能从 `Scheduler.add_request()` 直接
抛出导致 EngineCore 失败。完整 checkpoint/replay 仍为长期 backlog。

**v1 守卫条件**（bind/启动时逐项检查，不满足明确报错）：

```text
is_profiling=False；max_num_seqs>=1；enable_prefix_caching=False；
目标 group 为 FullAttentionSpec（非 sliding window/MLA）；kv_cache_dtype 非量化；
scheduler/kernel block size 相同且 page 无额外 padding；
无 KV transfer connector（正负共享层名，connector 会把负分支写入误记为正向层事件）；
PP=1；DCP=1 且 PCP=1（手工 metadata 不含 dcp_local_seq_lens 语义）；
use_ubatching=False（ubatch wrapper 会 override_forward_context(None) 且多线程 capture，
与嵌套 context 冲突）；enable_sleep_mode=False（负池 tensor 未注册进 sleep/wake 生命周期）；
speculative_config 为空；enforce_eager=True；
正池 >= max_num_seqs * max_model_len；
负池 blocks >= max_num_seqs * ceil(max_model_len / block_size)；
bind 期显存 pre-flight 通过
```

**v1 测试矩阵**：

```text
CPU   allocator append/reset/free；slot/block 数学；deferred free；N*max_model_len C1/C2
      容量守卫；双请求 negative 顺序 forward；same/different controls RNG 分组；多请求
      terminal drain 和 sparse waveform routing；未声明 capability 模型零行为变化
冒烟  真实 builder metadata 构造（版本漂移第一闸）
GPU   负分支逐次 append vs Transformers Qwen2 同权重朴素 forward，hidden 逐行 bf16 close；
      TP=2 两请求同时 resident；mixed prefill/decode；same-controls `[2B,64]`；同步
      terminal drain；abort one/survivor；双 rank allocator parity
E2E   Transformers PR #40546 三步 cached full-stack golden；真实 TP=2 HTTP uploaded voice/batch/SSE/WS；natural EOS、
      length cap、terminal drain；其他 AR TTS capability/header no-op 门禁
```

固定并发扩展必须补双/多请求 active-subset 和无抢占证明。swap 路径若经 profiling 立项，
再补 arena、victim、microbatch、bit-exact 换回、持续超卖吞吐和动态 scheduler 测试，不提前
混入 v1 matrix。

**演进路线（非 v1 范围）**：先增加 negative GPU pool 和固定 safe concurrency；随后才是
active-subset microbatch、CPU arena/swap、动态 reservation scheduler、负分支 CUDA graph、
负池并入 vLLM 内存核算，以及 opt-in embedding replay/checkpoint 灾难恢复。每一项都需要
独立 profiling 或第二个 capability 用户作为立项门槛。

**让本方案不再最优的条件**（定期复评）：vLLM 上游提供原生 side-channel KV 支持 ->
迁移；metadata fabrication 成为实测瓶颈 -> 增量缓存优化；第二个模型需要旁支 KV ->
泛化 NamedCausalKVBranch 并将 knob 提升为框架统一键。

### 12.9 风险登记表

| 风险                                                                                                                                                         | 当前决策/缓解                                                                                                                                                                                                                        | 后续归属     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| Encoder cache 泄漏                                                                                                                                           | 真实 EngineCore 已验证 finish/abort -> freeable -> 压力驱逐 -> GPU 删除                                                                                                                                                              | 完成         |
| 请求参数绕过 pipeline constraints                                                                                                                            | 正常 TTS 已 model-specific 收口；低层完整列表替换仍是高级 API 限制                                                                                                                                                                   | 已登记       |
| 用户非零 temperature 覆盖官方 argmax                                                                                                                         | deploy 默认与 VibeVoice adapter 最终 override 均固定为 0；对象/dict/幂等测试通过                                                                                                                                                     | 完成         |
| Batch-dependent VAE RNG                                                                                                                                      | 保持官方全局 RNG；UUID 派生 generator 为可选项                                                                                                                                                                                       | M4 决策      |
| Serving UUID 未注入                                                                                                                                          | Adapter 已生成 request/item-scoped UUID，并用 token-prompt 绕过 pinned text-path 丢 UUID 问题                                                                                                                                        | 完成         |
| 裸 2D audio 歧义                                                                                                                                             | 已拒绝，要求 tuple 或 list                                                                                                                                                                                                           | 完成         |
| Processor/MM 相关私有 vLLM API（`_get_audio_with_sr`、`_merge_multimodal_embeddings`、stage-0 tokenizer）                                             | 已收口到`vllm_compat.py` 并加 smoke test；negative KV 的接触面收口在 `named_kv_branch.py`，见下方独立风险行                                                                                                            | 升级门槛     |
| Ragged audio padding 成本                                                                                                                                    | 真实负载测量前不做排序/分桶                                                                                                                                                                                                          | Perf backlog |
| TP/小卡覆盖                                                                                                                                                  | TP=2 完整 negative CFG/M4a/M4b/waveform 与真实 AsyncOmni 已通过，默认 deploy 为 TP=2；最低显存仍待干净 H100 分阶段和可选 2×24GB profiling 定义                                                                                       | Pre-merge deployment profiling |
| Converted HF shard 缺失                                                                                                                                      | 不阻塞；自动 golden 从官方三 shards test-only 在线映射 HF state，无需持久转换 shard                                                                                                                                                  | 完成         |
| 通用 MM profiler 与对称 placeholder 校验冲突                                                                                                                 | 固定 KV bytes +`skip_mm_profiling=true`；独立真实上界测试兜底                                                                                                                                                                      | 完成         |
| Stateful CFG 需要第二套 PagedAttention KV                                                                                                                    | PR-1 store + PR-2 VibeVoice executor bind 已完成；official 两步/真实 Omni transition/TP=2 通过                                                                                                                                       | 完成（v1）   |
| 直接 vllm.LLM 使用上游 GPUModelRunner，不安装 Omni named-KV capability，也不执行 preprocess/postprocess/make_omni_output/stateful transition                | 实测 stock 探针（显式强制 control-token）只产控制 token、响应无 waveform；capability acknowledgement 会在日志中 warning_once，不再是完全静默。普通 stock 调用不经过 pipeline sampling_constraints，还可能采样完整词表——两种形态都无波形。低层路径仅保留 Processor/Encoder-cache/prefill 测试；M4c waveform 仅支持 AsyncOmni/GPUARModelRunner。`OmniGPUModelRunner.load_model`（先于 profile_run/dummy forward）只为声明 named-KV request 的模型置标志；合法 profiling 不误报，未声明模型零行为。支持 upstream runner 属于计划外，未修改 | 警告完成；支持已登记 |
| Transformers PR#40546 当前 checkout 缺少 `LOGITS_PROCESSOR_INPUTS_DOCSTRING` import，且本地 HF checkpoint 缺 shard 1                                       | 未修改外部 runtime；test-only helper 在隔离进程中注入缺失 doc constant，并从官方三 shards 在线映射出 HF state（1205 含 tied lm_head、missing/unexpected=0）。自动 golden 已运行 PR 原生 full cached `generate()`，并以真实 Omni condition/noise 重放 PR diffusion + cached decoder/Semantic feedback 做逐步有界对拍；无需生成持久转换 checkpoint | 自动回归完成；外部缺口已登记 |
| Stock runner 在下一次 scheduled forward 才消费 sampled `audio_token`；hard length cap 后没有下一次 forward | 已实现 capability-gated post-sample drain：bookkeeping 后仅声明模型检查 terminal token，先执行本步正常 postprocess，再由 VibeVoice 独立推进 negative Qwen 和 M4a/M4b，将最后 3200-sample chunk 合并进 snapshot 前 sparse output；EOS/stop 优先级不变。未声明模型零额外 D2H/forward/属性。官方 checkpoint、TP=2、AsyncOmni 已验证 3 tokens→9600 samples，negative blocks 全归还；strict xfail 与过期 serving warning 已移除 | 完成 |
| 父请求正 KV 被 scheduler 抢占                                                                                                                                | 正池 `max_num_seqs * max_model_len` 启动守卫已实现；默认两请求可完整常驻，使 KV 压力抢占不可达                                                                                                                                         | 完成         |
| 负池耗尽                                                                                                                                                     | 负池按 `max_num_seqs * ceil(max_model_len/block_size)` 启动守卫；默认 4 GiB/rank，运行时禁止超卖                                                                                                                                       | 完成         |
| 负池显存未纳入 vLLM 核算（固定 kv_cache_memory_bytes 跳过自动 sizing）                                                                                       | bind 期 pre-flight 已实现（free VRAM >= 负池 + 512 MiB 默认 activation margin）；预算合并核算列为演进                                                                                                                                | 完成（v1）   |
| 持续超卖下换出抖动                                                                                                                                           | v1 不超卖；swap/microbatch 只有 profiling 证明必要且定义背压门槛后才立项                                                                                                                                                             | Perf backlog |
| idle abort 后 side state 延迟释放                                                                                                                            | 有界（<= max_num_seqs 份），下一次完整 scheduled-set forward / final postprocess / 进程退出时释放；逐请求 preprocess 和 step 末 flush 在 async scheduling 下都不安全。free() 只归还 block ID，不擦除 KV tensor。test-only worker extension + collective RPC 已在真实 TP=2 双 resident 请求证明 abort one 后 survivor 继续，最终 Acoustic/Semantic/waveform state 删除且负池全部 block 归还 | 完成         |
| 负分支依赖 vLLM 私有接触面，主要至少包括：forward 路径语义（kv_cache 属性 / override_forward_context / create_forward_context / CommonAttentionMetadata / builder 构造与 build()，静默漂移风险，conformance 兜底）；bind 期结构读取（`_kernel_block_sizes` / `attn_groups` / `kv_cache_config` / `static_forward_context` / backend get_kv_cache_shape / get_kv_cache_stride_order / FullAttentionSpec page 与 layout 字段，上游改名即 bind 失败，响亮不静默） | 收口 named_kv_branch.py 单文件 + bind 冒烟 + conformance 测试；fake-runner conformance 伪造 runner 结构，真实 AsyncOmni 路径是 bind 期失效的检测闸；清单按“主要接触面”登记，不做精确计数                                                                                                                                | 升级门槛     |
| async scheduling 下 negative forward 异常后的排队 step                                                                                                       | 真实 TP=2 fault injection 证明首个异常已丢弃负 branch 并触发 EngineCore fatal；已排队 step 仍可能二次报 `must be reset before append`，scheduler 随后可见 request-index `KeyError`。最终 worker shutdown 已证明双 cache/model state 释放、branch close、allocator 全部 block 归还；其中一次测试宿主退出时 resource_tracker 报告并回收 1 个 leaked shared_memory object，后续隔离及全量重跑未复现。抑制 secondary error/若复现则修复 fatal shared-memory teardown 均需 shared async runtime 修改，计划外未实施 | 已登记       |
| 正常 AsyncOmni shutdown 偶发残留 process-manager 子进程                                                                                                      | focused full golden 的一次成功运行在两 TP worker graceful exit 后仍记录 `Process manager: force killing remaining processes count=1`；随后包含相同 golden 的 VibeVoice 全量成功且未复现。资源生命周期断言和测试结果不受影响；定位/修改 shared process manager 属于计划外 runtime 工作，先登记不实施                                                         | 已登记       |
| 负分支无 CUDA graph                                                                                                                                          | 继承 enforce_eager 现状，开销非首要（正 decode > M4a > 负分支）；VoxCPM2 decode-graph 路径可复用，profiling 门槛                                                                                                                     | Perf backlog |
| Diffusion steps 配置来源不同                                                                                                                                 | stateful hook 已消费 request/deploy`extra_args.num_diffusion_steps`；缺失时回退 model config                                                                                                                                       | 已接线       |
| Diffusion noise/RNG ownership                                                                                                                                | 同 control active subset 使用官方一次 global device RNG`[2B,64]`；自动 Transformers PR golden 以 deterministic test-only noise 对拍数值，生产仍保持全局设备 RNG；不同 control 分组及动态 batch ordering 仍会影响随机流                                                                 | 已验证；batch-order 风险已登记 |
| DPM scheduler NumPy 2 warning                                                                                                                                | Microsoft 和 diffusers 都有`np.array(torch_tensor)` deprecation，当前十步结果逐值一致                                                                                                                                              | Upstream     |
| BF16 semantic cached/full 不 bit-exact                                                                                                                       | H100 最大绝对差 0.125；官方 cached 路径为权威，测试与 Transformers PR cached full-stack golden 均保留 0.25 有界保护                                                                                                                   | 完成         |
| 跨 step condition alias runner 可复用 GPU buffer                                                                                                            | `_validate_condition()` 原先使用 `detach().contiguous()`；连续 input slice 不复制，导致 negative audio-BOS embedding 在下一步消费前被 runner `inputs_embeds` buffer 覆写。319 字生产探针出现 raw RMS 0.342、peak 1.75、1.91% 样本越过 ±1，PCM16 WAV 因硬 clipping 听感变大/模糊。现改为 request-owned `detach().clone(memory_format=contiguous_format)`，并对 positive/negative input/condition 统一所有权。修复后同探针 raw RMS 0.0626、peak 0.621、无越界；PR 对照 RMS 0.0508、peak 0.691、无越界。正式 golden 新增首个 negative input 与 PR audio-BOS exact、native negative condition 有界对拍 | 完成 |
| Conv padding cache 依赖 batch index                                                                                                                          | M4b 不拥有 cache；M4c 使用 per-request cache，动态 pack/unpack 延后优化                                                                                                                                                              | M4c          |

### 12.10 当前已完成和待执行模块

已完成：

```text
VibeVoiceProcessingInfo
VibeVoiceDummyInputsBuilder
VibeVoiceMultiModalDataParser
VibeVoiceMultiModalProcessor
vllm_compat private-API boundary
ModelRegistry + MULTIMODAL_REGISTRY
SupportsMultiModal
embed_multimodal(sample=True)
_get_audio_embeddings(sample=False/True parity path)
per-item crop
embed_input_ids MM merge
tokenizer metadata fallback
pipeline allowed/stop constraints
CPU/GPU Processor + runner contract tests
VibeVoiceTTSAdapter + speaker/audio/UUID prompt contract
model-specific SamplingParams object/dict replace + 幂等测试
真实 EngineCore same/different UUID cache 测试
finish/abort -> freeable -> pressure eviction -> hash 通知 -> GPU tensor 删除
四 token allowed mask 到真实 SamplingMetadata
TP=2 Qwen 分片 + side modules 复制/一致性
M4a model-local diffusion 数值核 + fresh DPM solver
Microsoft/diffusers betas、timesteps、十步 prev_sample 逐值一致
官方权重 BF16 CUDA CFG + DPM 完整 denoise loop
M4b model-local inverse scaling + Acoustic Decoder + Semantic feedback kernel
两套 causal cache 显式输入/输出 + 双 chunk 官方权重 BF16 CUDA decode
M4c phase 1 request-local state machine + has_preprocess/has_postprocess + deferred cleanup
显式 one-step positive/negative condition + active-subset batched M4a→per-request M4b 接线
PR-0 单请求 conformance：tiny 28-layer Qwen2 使用独立 builder + 手工 CommonAttentionMetadata，
     四步 cached execution 与 Transformers BF16 max_abs_diff=0.03125；每步 28 层 kv_cache
     属性和外层 ForwardContext 均恢复；未修改 shared runtime
PR-1 固定 GPU pool NamedCausalKVBranch + runner capability + additional_config 通道；
     正负 `max_num_seqs * max_model_len`、显存/模式守卫、allocator/reset/free、异常时
     整分支丢弃、runner shutdown close、未声明模型零行为均已实现；public reset/free 在
     active context 内拒绝，内部 `_free_unchecked`/best-effort fault cleanup 保证释放且不覆盖
     原始 forward 异常；tiny 28-layer store 十七步 cached hidden 跨 block 对拍和
     block/fault/active-context cleanup 通过
PR-2 VibeVoiceNegativeBranch + vibevoice.py/stateful.py 已接线；Protocol 收敛为
     reset_audio_segment/forward_step/free，多请求按序 negative forward，deferred cleanup
     统一释放 store；官方 checkpoint
     连续两步 negative hidden、真实 Omni 强制 audio-token M4a/M4b transition、TP=2 rank-local
     negative KV/聚合 hidden 一致性均通过
PR-3 output path：sparse request-scoped waveform output（mono CPU FP32 / 24 kHz、
     drain-once + OutputProcessor 按请求累积）；serving WAV 序列化；adapter temperature=0.0；
     TP=2 full stateful/waveform；CPU abort/exception cleanup；真实 AsyncOmni 强制连续
     transition 出 9600 samples finite（含 terminal drain）；Transformers PR #40546 三步 cached full-stack golden 已覆盖
     PR 原生 generate 和 Omni condition/noise 驱动的 PR cached M4a/M4b 逐步有界对拍
```

待执行顺序：

```text
历史回归基线（Next-4 后工作树）：vibevoice 全量曾记录 137 passed + 1 xfailed；
      terminal drain 的旧 strict xfail 已转为通过；dummy 证据产生的 4-speaker dynamic xfail
      已移除；当前 strict real-weight 4-speaker HTTP 测试已在 advanced-model run level 通过；
      Transformers PR #40546 三步 cached full-stack golden、M3a EngineCore 生命周期、官方 checkpoint、GPU conformance、
      TP=2、真实 AsyncOmni finish/abort/exception cleanup 复测通过；shared runner/output
      P1 基线 90 passed。P1-2 自然 EOS 与 P1-3 真实 OpenAI speech HTTP 已完成：短文本及
      双说话人分别走 1/2 个 audio segment，最终 audio_eos→eos/finish_reason=stop；WAV、PCM、
      data URL、file URL 及流式/controls/seed/URI/60 秒/第 5 个 speaker 的 4xx 均已覆盖；
      hard length cap 的 HTTP 200 响应携带 `X-Finish-Reason: length`。
Gate  其他真实 AR 模型 GPU 回归：VoxCPM2 与 Qwen3-TTS 为必测门禁，Voxtral-TTS 在
      权重可用时作为可选对照；本地 checkpoint 可分别通过 `VOXCPM2_TEST_MODEL`、
      `QWEN3_TTS_TEST_MODEL`、`VOXTRAL_TTS_TEST_MODEL` 注入。门禁同时断言未声明模型的
      length-cap terminal drain 是 no-op，非 VibeVoice speech 响应无
      `X-Finish-Reason`。当前 real online tests 已显式检查 response header；terminal-drain
      零行为由 shared-runner 单测覆盖，VoxCPM2/Qwen3-TTS offline gate 也已接入通用 test-only
      worker snapshot，显式检查无 terminal capability、无 named KV 和请求残留，待干净 GPU
      实跑确认。真实权重门禁必须显式携带 run level，不能依赖 `advanced_model` marker 自动切换：

```bash
VIBEVOICE_TEST_MODEL_ROOT=/path/to/models \
pytest --run-level advanced_model -q \
  tests/e2e/online_serving/test_vibevoice_tts.py \
  tests/model_executor/models/vibevoice/test_vibevoice_omni_ar_runtime_gpu.py

VOXCPM2_TEST_MODEL=/path/to/VoxCPM2 \
pytest --run-level advanced_model -q \
  tests/e2e/online_serving/test_voxcpm2_tts.py \
  tests/e2e/offline_inference/test_voxcpm2_tts.py

QWEN3_TTS_TEST_MODEL=/path/to/Qwen3-TTS-Base \
pytest --run-level advanced_model -q \
  tests/e2e/online_serving/test_qwen3_tts_base.py \
  tests/e2e/offline_inference/test_qwen3_tts_base.py
```

Run-1 本轮已增加 4-speaker natural HTTP smoke、其他 AR response-header isolation、
      concurrency trace 显存峰值采集和 F6 单 speaker 非阻断 runner；均未修改 inference
      runtime。在其他用户进程稳定 idle、显存固定为 48,090 MiB/rank 的 GPU 4/5 上执行了
      受控共享卡验证：真实权重 TP=2 direct runtime `4 passed`；全卡峰值约
      64.3 GiB/rank、最低剩余约 16.8 GiB/rank，测试后恢复 48,117 MiB 基线，原 PID 显存
      未上涨且未观测到 SM 活动。随后两次 4-speaker HTTP 运行因遗漏
      `--run-level advanced_model` 而被 fixture 注入 `load_format: dummy`；其 512/1024 token
      length 结果和约 16.2 GiB 最低剩余值均不是 checkpoint 证据，已撤回。HTTP fixture 现以
      `require_real_weights` 在启动前 fail-closed；修正后真实权重完整 HTTP 模块 `6 passed`
      （含 4-speaker strict stop），同 corpus lifecycle/allocator trace `7 passed`。新的真实权重
      运行全卡峰值约 64.89 GiB/rank、最低剩余约 16.19 GiB/rank；该共享卡总量只能作为
      clean-card profiling 前的参考。lifecycle trace 观测到
      3 个模型选择的 audio segment，所有 BOS/EOS 转换平衡且最终 EOS/cleanup 正常；segment 数不等同于输入
      speaker 数，内容覆盖仍进入 F6 多说话人评测。shutdown 反复可见
      `force killing remaining processes count=1`，但无 worker extension 的 HTTP 运行也会
      复现，因此归入 shared process-manager P2 robustness 调查。该共享卡结果不能替代干净
      GPU 门禁。VoxCPM2 随后在另一张受控共享卡完成真实权重 online `2 passed`、offline
      `2 passed + 1 skipped`（包内可选 voice-clone example 缺失）；mixed prefill/decode 的
      test-only isolation snapshot 通过，HTTP 仍无 VibeVoice terminal header。为使门禁在本地
      环境可复现，通用 E2E OpenAI client 已禁用 localhost proxy 继承，Whisper worker 在系统
      无 ffmpeg 时使用 imageio-ffmpeg，纯英文相似度不再强制导入 OpenCC。共享卡运行还发现
      `OmniRunner` 旧 cleanup 会按进程名扫描全宿主并尝试终止无关 EngineCore；test helper 已改为
      只清理 shutdown 前捕获的当前 pytest 进程树 descendants，禁止触碰其他任务。本机仍未发现
      Seed-TTS corpus 和固定 judge checkpoint cache，因此 F6 目前只完成
      client/dataset/SSE wiring、non-stop aggregate exclusion 与 CPU 测试。待取得足够独占
      显存和评测资产后继续 Qwen3-TTS standard deploy、干净卡 VibeVoice/VoxCPM2 复跑及 F6
      baseline。Qwen3-TTS standard deploy 在剩余约 33 GiB 的卡上受控试启动，两 stage 初始化
      峰值超过安全阈值，监控只终止本测试进程组；该路径仍待至少约 48 GiB 自由显存的干净卡。
Next-1 真实 AsyncOmni stateful finish/abort/exception cleanup 已完成：test-only
      worker_extension_cls + collective RPC 覆盖 TP=2；finish/abort 在下一安全请求后释放
      Acoustic/Semantic/waveform state 和全部 negative block，abort 后无新 payload；同步
      injected negative-forward exception 触发 fatal shutdown 后 branch close、allocator
      全归还、KV tensor 引用清空。async queued-step secondary error 单独登记，未改 runtime。
Next-2 终止边界 chunk 已根治：`GPUARModelRunner` 在 bookkeeping 后，仅对声明
      `terminal_sample_drain_token_ids` capability 的模型检查 hard length cap；若 sampled
      token 命中，则先用本步 live hidden states 执行正常 postprocess，再调用模型 drain，
      并在 sync/async output snapshot 前合并 sparse waveform。VibeVoice drain 独立执行最后
      一次 negative Qwen forward + M4a/M4b，不额外调度 positive LM step；EOS/stop 仍优先，
      未声明模型零额外 D2H/forward/capability 属性。官方 checkpoint、TP=2、AsyncOmni
      强制 3 个 audio token 已从 6400 修正为 9600 samples，strict xfail 与过期 serving
      warning 已移除，两个 rank 的 negative branch blocks 最终全部归还。
Next-3 小项：reset/free 外部入口与内部 fault cleanup 重构、capability acknowledgement
      均已完成；append 热路径持久 buffer（5 tensor/4 次 H2D per append，与未来 graph
      buffer 共用设计）保持 profiling-gated，未实施。
Next-4 Transformers PR #40546 三步 cached full-stack 自动化 golden 已完成：Microsoft 官方
      checkpoint、TP=2 AsyncOmni 三次完整 cached transition / 9600 samples 与 PR 原生 cached
      generate 同时执行；reference latent、TP rank trace、token/audio shape/finite 全覆盖；
      首个 negative input 必须与 PR audio-BOS embedding exact，native negative condition
      必须有界 close，防止 request state alias runner 可复用 input buffer；
      另以 Omni 实际 positive/negative conditions + deterministic noise 驱动 PR diffusion
      + Acoustic/Semantic cached decode，逐步对拍 latent/waveform/semantic/next embedding。
      PR native negative reset（tail-KV copy）与 Omni fresh named-BOS 表示不直接逐值比较；
      negative Qwen 本身继续由独立 17-step Transformers cached conformance 覆盖。
F3     固定 safe concurrency=2 已完成：4 GiB negative pool、N*max_model_len 双池守卫、
       TP=2 真实双 resident、mixed prefill/decode、同 controls `[2B,64]`、同步 terminal
       drain、abort one/survivor 与双 rank allocator parity 均通过。
Perf-1 profiling 后评估是否继续提高固定 safe concurrency。
Perf-2 只有固定容量不能满足目标时，另行设计 microbatch/swap/dynamic reservation；
       不作为 M4 waveform correctness 完成条件。
Pre-merge deployment profiling 门槛：先在干净 H100 分阶段记录，再仅在产品目标需要时以
       2×24GB 设备验证相同 TP=2 固定容量与安全余量。
```

## 13. 性能优化计划（已收敛，执行中）

### 13.1 基线与成本分解（H100，dummy 权重组件级实测，形状即真实值）

每个 audio token = 3200 samples @ 24 kHz ≈ 133 ms 音频。当前 eager 每 token 成本：

| 组件 | 实测 | 说明 |
|---|---|---|
| M4a decode（Acoustic Decoder + Semantic Encoder，B=1） | ~11.5 ms | 逐请求串行，B=2 时 ~23 ms，最大单点 |
| Diffusion 10-step CFG loop（B=2） | ~10.8 ms | 单次 head fwd 0.70 ms（tiny kernel launch-bound）；每次新建 DPM scheduler 0.57 ms |
| Negative Qwen（28 层 ×1 token/请求） | ~2 ms × 逐请求串行 | 另有每请求 metadata 重建、4 次小 H2D、56 次 `layer.kv_cache` 交换、forward_context 重建 |
| Positive Qwen decode（B<=2） | ~2-3 ms | eager；attention 已是 FA3 |
| preprocess `.item()` / waveform D2H | 各每请求每 step 一次 | `.item()` 处于 runner 已同步的 input-prep 区内，代价低于初估；D2H 是管线停顿点 |

合计 ~35-45 ms/token（B=2）→ RTF ≈ 0.26-0.34。目标：B=4 时 RTF <= 0.10（~12 ms/token）。

torch.compile 探针（仅参考，不作为路线）：diffusion loop reduce-overhead 4.2 ms
（加速 2.6x，但 bf16 latent 最大漂移 2.3e-2，贴 golden atol=0.03 边缘）；conv 栈
reduce-overhead 编译在本机 Triton launch 段错误（环境问题，已记录）。
**结论：图化走手动 CUDA graph capture（同 kernel 同顺序、逐位一致），不走 inductor。**

### 13.2 性能指标口径（复用仓库现有设施，不另造轮子）

- 主指标：audio_rtf p50/p95（B=1 与 B=4 两组，`compute_audio_rtf`，SLO < 1.0）、
  TTFA（`serving_time_to_first_output_ms`，含 reference prefill + 首 token 全链路）、
  audio_throughput（聚合音频时长/wall time）。
- 流式健康度：audio_continuity_ok rate（SSE，0.1 s underrun 阈值）。
- 门禁指标：per-token 阶段耗时分解（Phase A harness 输出，基线与每次对比写入本节）。
- 质量不变式：全部 golden/parity 测试 + 真实权重 HTTP 6 项 + lifecycle 7 项 + F6 指标不退化。

### 13.3 图粒度决策：分段图，不做单个 full graph（已收敛）

full graph（negative→diffusion→M4a→splice→positive 一张图）相对分段图的增量只有
子图间 ~0.2-0.5 ms CPU glue（`async_scheduling=True` 下基本被隐藏，GPU kernel 序列相同），
但引入五个更硬障碍：① 活性子集 2^B 组合爆炸（padding 会破坏官方 RNG draw 形状并污染
非活性请求的 conv cache，语义错误）；② TP=2 下必须手工 capture all-reduce（分段方案中
手工 capture 的 diffusion/M4a 均无集合通信，positive 走 vLLM 官方 capture）；③ positive
attention metadata 持久化机制需重写 mini-runner（vLLM 已为 FULL decode 解决，粒度是
model.forward，无法包含 forward 外的过渡逻辑）；④ prefill/segment 边界/terminal drain
仍走 eager，full graph 需维护两条逐位一致的语义路径；⑤ 违反 vllm_compat 升级隔离原则。
**capture 边界画在语义分支点和外部依赖点上**：negative forward（共享 Qwen 层对象、TP
集合通信、动态 metadata）与活性子集分支恰好是组件图边界。
演进通道保留：若 C 阶段后实测 glue 占可见时间 >10%，可将 negative batched forward 独立
图化（需先调研 FA3 decode metadata host-side 依赖），仍不走 mega graph。

### 13.4 阶段计划与状态

| 阶段 | 内容 | 预期 | 状态 |
|---|---|---|---|
| A | env-gated 计时 harness；P5 微优化包：DPM scheduler 按 steps 缓存（先验证 reset 语义）、waveform pinned 非阻塞 D2H + event、negative append 持久 buffer；基线 RTF/TTFA | -2~4 ms | 执行中 |
| A+ | max_num_seqs 2→4（独立 config 变更；KV 容量已核算：4×65536×14 KiB/rank=3.5 GiB <= positive 6 GiB / negative 4 GiB，startup guard 自动复核） | 吞吐 x2 | 待执行 |
| B | negative branch 批处理：`append_and_enter_batch`，一次 metadata build + 一次 kv_cache 交换 + 一次 B-token varlen decode forward；顺带消除双重 clone | -2~4 ms (B=2) | 待执行 |
| C1 | diffusion loop 手动 CUDA graph：schedule GPU 常驻（按 steps 缓存）、`cond_proj(condition)` 外提（10 步不变，逐位一致）、guidance_scale 张量化（避免烘焙常量）；graph key=(B_active, steps)；连续 2 token graph-vs-eager 逐位对照验收 | -6 ms | 待执行 |
| C2 | M4a per-slot 手动 CUDA graph：per-slot 预分配 conv cache shim（只读反射收集 HF 各 conv 层 in_channels/left_pad，不改 vendored 代码）、segment 开始 `zero_()` 原地复位、audio_out 静态 buffer 接 pinned D2H | -15 ms (B=2) | 待执行 |
| C3 | `preprocess_finalize` capability-gated hook（逐请求 preprocess 循环后、capture 区前）+ positive FULL decode graph；`NamedCausalKVBranch` 放宽 enforce_eager 仅限声明该 hook 的模型 | -1~2 ms | 待执行 |
| D | 可选独立 PR：4-token gate lm_head GEMV（467 MB->12 KB 权重读）、M4a 跨请求 batch、negative stream overlap（需 TP 集合通信分析）、reference encoder 编译、negative forward 独立图化调研 | 各 0.1-2 ms | 待评估 |

约束（用户确认）：不修改计划之外的 runtime 代码；发现计划外问题只记录在本节，不就地修复。
所有新路径必须可经 runtime config 回退 eager；每阶段门禁 = 全部现有 golden/parity + 真实权重
HTTP/lifecycle + RTF/TTFA 对比写入本节。

### 13.5 执行日志

- Phase A 完成（真实权重，TP=2，GPU 6/7 受控共享卡，B=1，SSE 长文本）：
  - 基线：RTF=0.436，TTFA=3241ms，232 tokens / 30.9s 音频 / 13.5s 总；每 token ~58ms（含 SSE/RPC）。
  - 阶段分解（TP0 稳态 188-token 均值，level=1 CPU 计时，async scheduling 下为 enqueue 口径）：
    - transition（diffusion+m4a+splice）= 24.2ms
    -   其中 diffusion=10.8ms（与组件微基准一致）、m4a_decode=13.2ms（含逐请求循环开销）
    - negative_forward=8.5ms（逐请求串行 + 每请求 metadata build + context manager）
    - positive_forward=11.8ms（远高于组件微基准 ~2ms：28 层 FA3 decode 的每层 metadata build + Python + kernel launch 占主导，非 matmul 本身）
    - preprocess=1.6ms（逐请求 Python 循环 + `.item()` sync，处于 runner 已同步的 input-prep 区内）
    - postprocess=0.03ms（可忽略）
  - 关键修正：positive_forward 11.8ms 是第二大成本（仅次于 transition 24.2ms），
    根因是 eager decode 的每层 FA3 metadata build + Python launch 开销，
    而非 lm_head 全 vocab GEMM（后者仅 ~140µs，memory-bound）。因此 **C3（positive FULL decode graph）收益远高于初估**，
    它消除每层 metadata build 与 launch 开销；D2（4-token gate lm_head GEMV）仅省 ~140µs，降级。
  - Phase A 微优化已落地：DPM scheduler 按 steps 缓存（逐位一致，已加 parity 测试）、
    waveform pinned 非阻塞 D2H + event 回收、negative append 常量 query_start 持久化、
    env-gated 计时 harness（`VLLM_OMNI_VIBEVOICE_PERF_TIMING`，默认关闭零开销）。
  - 修正记录：preprocess `.item()` 处于 runner `synchronize_input_prep` 区内，代价低于初估，
    Phase A 不改 shared runner（避免计划外 runtime 改动），留到 C3 的 `preprocess_finalize` 一并处理。
  - 既有 lint 修复（计划外但 test-only、零 runtime 风险）：`test_vibevoice_processing.py` I001、
    `test_vibevoice_negative_kv_conformance_gpu.py` TID251（`torch.cuda.empty_cache`→`torch.accelerator.empty_cache`）。
  - CPU 回归：175 passed / 5 skipped（VibeVoice 全量 CPU 套件）。
  - 真实权重 correctness：baseline 测试 `1 passed`（finish=stop，232 tokens，finite audio）。
- Phase A+（max_num_seqs 2→4）完成：
  - yaml `max_num_seqs=4` + config 断言同步；KV 容量由 startup guard 自动复核通过
    （negative 实测 18,724 blocks / 299,584 tokens ≥ 4×65,536=262,144）。
  - HTTP 4 并发门禁 `6 passed`；B=1 基线复测 RTF=0.431（与 Phase A 一致）。
  - 4 并发长文本 SSE：4/4 finish=stop，per-request RTF 0.86-0.91，wall=30.3s，
    total_audio=132.0s，**aggregate_x=4.35**（B=1 单流为 2.32x）。
  - B=4 阶段分解（441 transition 均值）：diffusion=11.3ms（跨请求已 batch，平稳）、
    m4a_decode=13.5ms×逐请求串行、**negative_forward=22.3ms**（~2.4 活性请求×串行，
    Phase B 目标）、positive_forward=9.4ms（vLLM 已 batch）。
  - 记录的问题（计划外，不就地改 runtime）：serving 层抓取用户 URL 时继承宿主代理
    env，错误信息与可达性依赖宿主代理状态（`rejects_invalid_requests` 首跑因此 flake）；
    留待 serving robustness 工作评估。测试侧修复（test-only）：测试服务器 env 注入
    `NO_PROXY=127.0.0.1,localhost`，localhost 抓取错误信息确定后复跑全绿。
  - 记录的既有 lint 问题（计划外，不就地修复）：GPU 测试文件
    `test_vibevoice_processing_gpu.py`（I001、TID251×5、**F821 undefined name `model`——
    真实测试 bug，执行到该行会 NameError**）、`test_vibevoice_tp2_gpu.py`（TID251×3）、
    `test_vibevoice_weight_loading_gpu.py`（TID251×1）在原始提交中即未过 ruff；
    属独立测试清理工作，不影响 runtime。
- Phase B（negative batched forward）完成：
  - `NamedCausalKVBranch.append_and_enter_batch`：一次批量 slot bookkeeping（先全量校验再
    变更，bookkeeping 失败 fault-free 整个 logical batch）、一次 metadata build（堆叠
    block table、arange query_start、批量 seq_lens/positions/slot_mapping）、一次
    kv_cache 交换、一次 B-token varlen decode forward。原 `append_and_enter` 保留，
    与批量路径共享 `_append_slots`。
  - `VibeVoiceNegativeBranch.forward_step` 改为单次批量 forward；批量 clone 一次后
    unbind 行视图（消除逐请求双重 clone）；校验逻辑不变。
  - 数值门禁：conformance GPU 测试新增批量段——两个错位请求（batch-a 先行 4 步）共享
    批量 attention context，逐行对拍独立 HF cached 参考，跨 16-token 页边界，
    batch_max_abs_diff ≤ 0.04 通过；17-step 单请求段保持通过。
  - 全量门禁：CPU 177 passed；TP2 lifecycle 8 passed；HTTP 4 并发 6 passed。
  - 性能（B=4，441 transition 均值）：**negative_forward 22.3→9.9ms（-55%）**；
    C4 聚合吞吐 **4.35→5.60 audio-s/wall-s（+29%）**；C4 per-request RTF
    0.86-0.91→0.66-0.70；4 并发 wall 30.3→23.0s。B=1 RTF 0.479（共享卡基线噪声内）。
- Phase C1（diffusion loop 手动 CUDA graph）完成：
  - `VibeVoiceDiffusionGraphExecutor`：手动 capture 整个 10-step DPM loop（含
    `cond_proj` 外提——逐位一致），graph key=`(B_active, steps, guidance)`；
    guidance 保持 Python float 烘焙（device 标量会 bf16 舍入 1.3，无法逐位复刻
    eager 的 python-float 乘法语义）。capture 失败永久回退 eager。
  - 关键技术发现：diffusers 在 **CPU 上算 schedule 标量**（log/exp），CPU 与 GPU
    超越函数 kernel 非同舍入——把 sigmas 移到 GPU 会引入 ~0.03 bf16 漂移（> golden
    atol=0.03）。解法：schedule 留 CPU；0-dim CPU 标量被 GPU elementwise op 消费时
    **作为 kernel 参数烘焙**（无 H2D），replay 复用 capture 时的精确值（schedule
    按 steps 恒定）。唯一 capture 非法 op 是每步 timestep 的 `.to()` H2D——
    capture 前预计算 10 个 GPU timestep_batch 一次解决。无需 pinning mode、
    无需重写 solver。
  - 数值门禁：graph-vs-eager 逐位对照（B∈{1,2,4}、连续 token、guidance/steps 变体、
    capture 失败回退）5 passed；golden 3-step cached PR parity 7 passed/1 skipped。
  - 性能（B=4，441 transition 均值）：**diffusion 11.3→3.2ms（-72%）**；
    B=1 RTF 0.431→0.343（-20%）；B=4 聚合 5.60→6.01 audio-s/wall-s；
    transition 44→36ms。m4a_decode（13.3ms）与 negative_forward（9.7ms）不变
    （分别是 C2 与已完成的 B 的目标）。
  - runtime config 开关 `diffusion_cuda_graph: bool = True`（默认开，可回退 eager）。
- Phase C2（M4a per-slot graph）：预期 13.2→~3-4ms。
- Phase C3（positive FULL decode graph）：预期 11.8→<2ms（收益上修）。
- 预期合计：~44ms → ~12-15ms/token，RTF 0.44→~0.10。
