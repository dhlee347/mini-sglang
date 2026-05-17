# minisgl.models — LLM 모델 구현

## 역할

지원하는 LLM 아키텍처를 구현하고, HuggingFace 체크포인트에서 가중치를 로드·샤딩합니다.

**파일**:
- `base.py` — `BaseLLMModel` 추상 클래스
- `config.py` — `ModelConfig` (HF config → 내부 config 변환)
- `register.py` — 아키텍처명 → 모델 클래스 레지스트리
- `weight.py` — 가중치 스트리밍 로더 (샤딩, 병합, MoE 패킹)
- `utils.py` — 모델 생성 헬퍼
- `llama.py` — LLaMA-3
- `qwen3.py` — Qwen3 dense
- `qwen3_moe.py` — Qwen3 MoE
- `qwen2.py` — Qwen2.5
- `mistral.py` — Mistral

---

## `base.py` — `BaseLLMModel`

```python
class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...
    # 입력 없음 — Context.batch를 통해 전역 접근
    # 출력: logits [seq_len, vocab_size]
```

모델 포워드는 인자를 받지 않고 `get_global_ctx().batch`에서 `input_ids`, `positions`를 읽습니다.

---

## `config.py` — `ModelConfig`

HuggingFace config를 파싱해 아키텍처 독립적인 내부 표현으로 변환합니다.

```python
hidden_size: int
num_layers: int
num_qo_heads: int
num_kv_heads: int
head_dim: int
vocab_size: int
intermediate_size: int
rotary_config: RotaryConfig
is_moe: bool
num_experts: int | None
num_experts_per_tok: int | None
```

**`RotaryConfig`**: `max_position`, `base`, `rotary_dim`, `scaling` (YaRN, Linear 등)

---

## `register.py` — 모델 레지스트리

```python
_MODEL_REGISTRY = {
    "LlamaForCausalLM":                   (".llama",      "LlamaForCausalLM"),
    "Qwen2ForCausalLM":                    (".qwen2",      "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM":                    (".qwen3",      "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM":                 (".qwen3_moe",  "Qwen3MoeForCausalLM"),
    "MistralForCausalLM":                  (".mistral",    "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration":    (".mistral",    "MistralForCausalLM"),
}
```

`get_model_class(architecture, config)`: 아키텍처 문자열로 레지스트리 조회 → lazy import → 인스턴스 생성.

새 모델 추가: 새 파일 작성 후 이 레지스트리에 한 줄 추가.

---

## `weight.py` — `load_weight()`

safetensors 파일에서 가중치를 스트리밍 로드합니다. 피크 CPU 메모리를 최소화하기 위해 텐서 단위로 처리합니다.

### 처리 단계

**1. TP 샤딩** (`_shard_tensor`):

| 가중치 타입 | 샤딩 방향 |
|-------------|-----------|
| `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` | dim=0 분할 (Column Parallel) |
| `o_proj`, `down_proj` | dim=1 분할 (Row Parallel) |
| `lm_head`, `embed_tokens` | vocab dim=0 분할 |
| 그 외 | 복제 (replicated) |

`k_proj`, `v_proj`에서 `num_kv_heads < tp_size`이면 head 단위로 분배.

**2. 프로젝션 병합** (`_get_merge_info`):

체크포인트의 분리된 가중치를 모델이 기대하는 fused 형태로 병합:
- `q_proj + k_proj + v_proj` → `qkv_proj`
- `gate_proj + up_proj` → `gate_up_proj`

`merge_buf` 딕셔너리에 파트별 수집 후 모두 모이면 `torch.cat` 후 yield.

**3. MoE 전문가 스택** (`_get_expert_stack_info`):

`model.layers.N.experts.M.gate_proj` 형식의 키를 `model.layers.N.experts.gate_up_proj` 형식으로 변환하고, M번 전문가들을 모두 수집 후 `torch.stack(experts, dim=0)`으로 패킹.

```python
# 체크포인트
experts.0.gate_proj   shape: [out, in]
experts.1.gate_proj   shape: [out, in]
...
# 런타임 (stacked)
experts.gate_up_proj  shape: [num_experts, out, in]
```

---

## 모델 구조 패턴 (LLaMA 예시)

```python
class LlamaForCausalLM(BaseLLMModel):
    embed_tokens: VocabParallelEmbedding
    layers: OPList[LlamaDecoderLayer]
    norm: RMSNorm
    lm_head: LinearReplicated

    def forward(self) -> Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        x = embed_tokens(batch.input_ids)    # 토큰 임베딩
        for layer in layers:
            x = layer.forward(x)             # Transformer 레이어
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)]  # 마지막 토큰만
        x = norm(x)
        return lm_head(x)                    # logits
```

decode 시: 모든 토큰이 마지막 토큰이므로 별도 슬라이싱 불필요.
prefill 시: `get_last_indices()`로 각 요청의 마지막 토큰 위치를 가져와 슬라이싱.

---

## 다른 모듈과의 관계

```
Engine.__init__
    → create_model(config.model_config)  → register.get_model_class()
    → model.load_state_dict(load_weight(model_path, device))

BaseLLMModel.forward()
    → get_global_ctx()                   → core.Context
    → layers[i].forward()                → layers.AttentionLayer
                                         → attention 백엔드
    → ctx.batch.attn_metadata.get_last_indices()
```
