
# VibeVoice-TTS 官方 checkpoint 适配说明

本文记录非 Realtime VibeVoice-TTS 在 vLLM-Omni 中的配置契约。公开输入是 Microsoft 官方原始 checkpoint；运行时统一使用 Transformers PR #40546 的 HF schema，以便复用已合入的 Acoustic Tokenizer 和参考完整 TTS PR。

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
当前限制为每条
60 秒（450 placeholders），每请求最多 8 条；deploy YAML 同时显式设置
`limit_mm_per_prompt.audio = 8`。最坏 reference prompt 使用 `8 * 450 = 3600`
MM tokens，加上文本后仍低于 65,536 context limit；8,192 的 per-iteration budget
通过 chunked prefill 分批调度。

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
| M4c stateful AR integration    | 进行中（PR-0/PR-1/PR-2 与 PR-3 output path 完成；执行计划见 §12.8） | 真实 Omni audio-token transition 与 waveform payload 通过；待 Microsoft full-generation golden 和异常压测 |

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

低层 `LLM.generate()` 是高级 API，调用者必须显式提供 request-scoped UUID；Processor
不得隐式生成随机 UUID，否则会破坏同请求 chunked-prefill 复用。完整 waveform E2E
延后到 M4。

### 12.8 M4 门槛和 golden reference

TP=2 gate 已完成：vLLM Qwen2 的 packed QKV/gate-up 按 rank 分片，Acoustic/Semantic
Encoder、projector、Diffusion Head 和 latent scale/bias 在各 rank 复制；相同输入的
projector/diffusion 输出逐值一致。覆盖位于 `test_vibevoice_tp2_gpu.py`。

M4 完成前仍必须：

- 明确最低 GPU 显存支持，或增加小卡完整 engine profiling 测试；
- 验证 final step 可读 per-request state，随后 finish/abort/exception 无状态泄漏；
- 使用 Microsoft 官方仓库实现作为 decode E2E golden。M2 已证明 PR prefill 公式等价，
  缺失的 converted HF shard 不阻塞关键路径。

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
后续 Microsoft golden 必须比较相同 cached execution，不能把 full-sequence bit equality
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

修订记录：v3 保留 runner-owned NamedCausalKVBranch 最终 ownership，但将实施顺序调整为
correctness-first：v1 固定 `tensor_parallel_size=2`、`max_num_seqs=1`、每 rank 2 GiB
negative GPU pool、无 swap、无自定义 scheduler，先完成 Microsoft waveform E2E；随后
优先以增大 GPU pool 的方式扩展固定安全
并发。active-subset microbatch、CPU arena、swap 和动态 reservation scheduler 不再是首版
依赖，只有 profiling 证明固定容量无法满足目标负载时才实施。D1 继续使用
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
    negative_kv_cache_memory_bytes: 2147483648       # 2 GiB，缺省同值
    negative_kv_activation_margin_bytes: 536870912   # 512 MiB，缺省同值
```

解析侧新增 `models/vibevoice/runtime_config.py`（类型化 dataclass，仿
`voxcpm2/runtime_config.py`：默认值、coerce、未知键 warning、下限校验），读取
`vllm_config.additional_config["vibevoice_runtime_config"]`。容量公式（TP=1）：
28 层 x 2(K,V) x 2 kv heads x 128 x 2 B = 28,672 B/token；2 GiB ≈ 74.9k negative
tokens；v1 以 `max_model_len=65,536` 做单分支启动守卫，不分配 swap arena。deploy
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
C1 单请求负分支可容：
     negative_pool_tokens >= max_negative_sequence_tokens
     max_negative_sequence_tokens 使用 max_model_len=65,536 作为低层请求绝对上界，
     不能只依赖 deploy 默认 max_tokens=40,500
     当前 2 GiB 估算容量约 74.9k token，因此单请求成立
C2 单请求正分支可容：
     positive_pool_blocks >= ceil(max_model_len / block_size) + 必要 headroom
     当前 6 GiB 估算容量约 224k token，因此单请求成立
C3 首版并发固定：
     tensor_parallel_size=2（默认 deploy；side modules 每 rank replicated）
     max_num_seqs=1
     正负两池均可完整容纳唯一 running request，KV 压力抢占构造性不可达
```

任一守卫不满足都在启动期明确失败；运行时不以 forward 异常表达普通容量不足。由于
单请求的正负完整 working set 都常驻 GPU，v1 不需要 victim、arena、microbatch 或
embedding replay，也不改变官方 global-device RNG 语义。

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
下一请求的 preprocess 或进程退出才释放——占用有界（<= max_num_seqs 份），不影响
正确性。不能改为 step 末无条件 flush：async scheduling 下 finish 判定晚一拍，step 末
无法知道该请求是否已被下一步调度，提前 free 会破坏 finished-and-scheduled 的最后
一次 forward；next-preprocess flush（带 exclude）是当前已知信息下最早的安全点。

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
    + bind 期实际容量/显存 pre-flight + 1 请求 metadata 冒烟
vllm_omni/worker/gpu_model_runner.py（纯增量、默认关闭）
  initialize_kv_cache 覆写：super() 后 _maybe_bind_named_kv_branches()
  （getattr(model, "named_kv_branch_request", None) 为 None -> 零行为变化；
  is_profiling 跳过；逐项守卫校验）
models/vibevoice/negative_branch.py（新增）
  VibeVoiceNegativeBranch：单请求 Protocol 实现
models/vibevoice/runtime_config.py（新增，D1，读 vllm_config.additional_config）
models/vibevoice/vibevoice.py
  named_kv_branch_request 属性、bind_named_kv_branch、删 on_requests_finished 转发
models/vibevoice/stateful.py
  Protocol 收敛、bind_negative_branch、cleanup_request 调 branch.free
models/vibevoice/pipeline.py / deploy/vibevoice.yaml
  继续使用 OmniARAsyncScheduler；v1 固定 max_num_seqs=1 和 negative pool 配置
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
自定义 scheduler 修补该问题，而是通过 `max_num_seqs=1` 和 C2 启动守卫保证唯一请求的
完整正序列可常驻 positive pool，使 KV 压力抢占构造性不可达。仍使用现有
`OmniARAsyncScheduler -> GPUARModelRunner`，若 deploy 覆盖 `max_num_seqs>1` 则启动失败，
不能只 warning。

**并发扩展**：优先把正负 GPU pool 同时扩大到固定安全并发，并继续使用现有 scheduler。
只有需要允许 `max_num_seqs` 大于 worst-case safe concurrency 时，才设计模型私有动态
reservation scheduler；届时必须按每个 admitted request 的完整 block 上界记账，不能只算
剩余增长，也不能仅覆写 `_should_defer_waiting_admission()`。请求级拒绝必须落在具有
request-local error handling 的 input/preprocess 层，不能从 `Scheduler.add_request()` 直接
抛出导致 EngineCore 失败。完整 checkpoint/replay 仍为长期 backlog。

**v1 守卫条件**（bind/启动时逐项检查，不满足明确报错）：

```text
is_profiling=False；max_num_seqs=1；enable_prefix_caching=False；
目标 group 为 FullAttentionSpec（非 sliding window/MLA）；kv_cache_dtype 非量化；
scheduler/kernel block size 相同且 page 无额外 padding；
无 KV transfer connector（正负共享层名，connector 会把负分支写入误记为正向层事件）；
PP=1；DCP=1 且 PCP=1（手工 metadata 不含 dcp_local_seq_lens 语义）；
use_ubatching=False（ubatch wrapper 会 override_forward_context(None) 且多线程 capture，
与嵌套 context 冲突）；enable_sleep_mode=False（负池 tensor 未注册进 sleep/wake 生命周期）；
speculative_config 为空；enforce_eager=True；
正池 >= 单请求完整 max_model_len；负池 >= 单请求完整 negative 上界；
bind 期显存 pre-flight 通过
```

**v1 测试矩阵**：

```text
CPU   allocator append/reset/free；slot/block 数学；deferred free；实际 C1/C2 容量守卫；
      未声明 capability 的模型零行为变化；故障注入后 28 层 kv_cache/context 必然恢复
冒烟  真实 builder 1 请求 metadata 构造（版本漂移第一闸）
GPU   负分支逐次 append vs Transformers Qwen2 同权重朴素 forward，hidden 逐行 bf16 close；
      positive/negative condition 对齐；长序列接近容量上界；TP=2 每 rank 实际布局/容量
E2E   单请求 TTS 出音频；finished-and-scheduled/abort cleanup；Microsoft 官方 cached golden
      对拍；旧 waveform-mislabel 与 temperature strict xfail 已移除；新增 hard max-token
      terminal audio-token 边界 xfail 见风险表
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
| 私有 vLLM API                                                                                                                                                | 已收口到`vllm_compat.py` 并加 smoke test                                                                                                                                                                                           | 升级门槛     |
| Ragged audio padding 成本                                                                                                                                    | 真实负载测量前不做排序/分桶                                                                                                                                                                                                          | Perf backlog |
| TP/小卡覆盖                                                                                                                                                  | TP=2 完整 negative CFG/M4a/M4b/waveform 与真实 AsyncOmni 已通过，默认 deploy 为 TP=2；最低显存仍待 profiling 定义                                                                                                                    | Pre-M4       |
| Converted HF shard 缺失                                                                                                                                      | 不阻塞；M4 使用官方实现作为 golden                                                                                                                                                                                                   | M4           |
| 通用 MM profiler 与对称 placeholder 校验冲突                                                                                                                 | 固定 KV bytes +`skip_mm_profiling=true`；独立真实上界测试兜底                                                                                                                                                                      | 完成         |
| Stateful CFG 需要第二套 PagedAttention KV                                                                                                                    | PR-1 store + PR-2 VibeVoice executor bind 已完成；official 两步/真实 Omni transition/TP=2 通过                                                                                                                                       | 完成（v1）   |
| 直接 vllm.LLM 使用上游 GPUModelRunner，不安装 Omni named-KV capability                                                                                       | 低层路径保留 Processor/Encoder-cache/prefill 测试；M4c waveform 仅支持 AsyncOmni/GPUARModelRunner；支持 upstream runner 属于计划外，不修改                                                                                           | 已登记       |
| Transformers PR#40546 当前 checkout 缺少 `LOGITS_PROCESSOR_INPUTS_DOCSTRING` import，且本地 HF checkpoint 缺 shard 1                                       | 未修改外部 runtime；临时进程内补 import、由官方 shards 在线映射出临时 HF state 后，人工 full`generate()` 三 token/9600 samples 通过；自动回归继续使用 Processor/DPM/cached M4b/official weights 分层 golden                        | 已登记       |
| Stock runner 在下一次 scheduled forward 才消费 sampled`audio_token`；若 `max_tokens` 正好结束于 audio token，最后一个 chunk 没有 final forward 可 decode | 正常 EOS-only 生成会在 EOS 前的 forward 解码已有 audio token；硬 max-token 截断与 Microsoft 即时 decode 差一个 3200-sample chunk。修复需要 post-sample model hook 或私有 sampler，属于计划外 runtime 修改，strict xfail 固定，不修改 | 已登记       |
| 父请求正 KV 被 scheduler 抢占                                                                                                                                | max_num_seqs=1 + 正池完整 max_model_len 启动守卫已实现，使 v1 KV 压力抢占不可达                                                                                                                                                      | 完成（v1）   |
| 负池耗尽                                                                                                                                                     | 负池可容完整 max_model_len + max_num_seqs=1 启动守卫已实现；运行时禁止超卖                                                                                                                                                           | 完成（v1）   |
| 负池显存未纳入 vLLM 核算（固定 kv_cache_memory_bytes 跳过自动 sizing）                                                                                       | bind 期 pre-flight 已实现（free VRAM >= 负池 + 512 MiB 默认 activation margin）；预算合并核算列为演进                                                                                                                                | 完成（v1）   |
| 持续超卖下换出抖动                                                                                                                                           | v1 不超卖；swap/microbatch 只有 profiling 证明必要且定义背压门槛后才立项                                                                                                                                                             | Perf backlog |
| idle abort 后 side state 延迟释放                                                                                                                            | 有界（<= max_num_seqs 份），下一请求 preprocess 或进程退出时释放；step 末 flush 在 async scheduling 下不安全，不为此改代码                                                                                                           | 已登记       |
| 负分支依赖 4 处 vLLM 私有接触面（kv_cache 属性 / override_forward_context / CommonAttentionMetadata / builder 签名）                                         | 收口 named_kv_branch.py 单文件 + bind 冒烟 + conformance 测试                                                                                                                                                                        | 升级门槛     |
| 负分支无 CUDA graph                                                                                                                                          | 继承 enforce_eager 现状，开销非首要（正 decode > M4a > 负分支）；VoxCPM2 decode-graph 路径可复用，profiling 门槛                                                                                                                     | Perf backlog |
| Diffusion steps 配置来源不同                                                                                                                                 | stateful hook 已消费 request/deploy`extra_args.num_diffusion_steps`；缺失时回退 model config                                                                                                                                       | 已接线       |
| Diffusion noise/RNG ownership                                                                                                                                | 同 control active subset 使用官方一次 global device RNG`[2B,64]`；不同 control 分组及动态 batch ordering 仍会影响随机流                                                                                                            | M4 golden    |
| DPM scheduler NumPy 2 warning                                                                                                                                | Microsoft 和 diffusers 都有`np.array(torch_tensor)` deprecation，当前十步结果逐值一致                                                                                                                                              | Upstream     |
| BF16 semantic cached/full 不 bit-exact                                                                                                                       | H100 最大绝对差 0.125；官方 cached 路径为权威，测试保留 0.25 有界保护                                                                                                                                                                | M4 golden    |
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
     max_num_seqs=1、正负完整 max_model_len、显存/模式守卫、allocator/reset/free、异常时
     整分支丢弃、runner shutdown close、未声明模型零行为均已实现；tiny 28-layer store
     十七步 cached hidden 跨 block 对拍和 block/fault cleanup 通过
PR-2 VibeVoiceNegativeBranch + vibevoice.py/stateful.py 已接线；Protocol 收敛为
     reset_audio_segment/forward_step/free，deferred cleanup 统一释放 store；官方 checkpoint
     连续两步 negative hidden、真实 Omni 强制 audio-token M4a/M4b transition、TP=2 rank-local
     negative KV/聚合 hidden 一致性均通过
```

待执行顺序：

```text
PR-3  已完成真实 24 kHz waveform output channel、单请求真实 Omni 多步 transition、serving
      WAV 序列化、adapter temperature=0.0、TP=2 full stateful/waveform、CPU abort/exception
      cleanup；Microsoft full cached generation 已人工确认同 prompt 的三 token 均为 audio、
      9600-sample waveform finite（现有分层 golden 仍是自动回归权威）。待真实 Engine abort
      压测和最低显存 profiling，无自定义 scheduler。
Perf-1 profiling 后优先增大 negative GPU pool并提高固定 safe concurrency。
Perf-2 只有固定容量不能满足目标时，另行设计 microbatch/swap/dynamic reservation；
       不作为 M4 waveform correctness 完成条件。
Pre-M4 最低显存/profiling 门槛。
```
