# minisgl.layers — 레이어 구성 요소

## 역할

Tensor Parallelism을 지원하는 LLM의 기본 빌딩 블록을 구현합니다. 모든 레이어는 `BaseOP`를 상속하며 자체 state_dict 직렬화/역직렬화를 구현합니다.

**파일**:
- `base.py` — `BaseOP`, `StateLessOP`, `OPList` (가중치 관리 프레임워크)
- `linear.py` — TP 지원 Linear 레이어들
- `attention.py` — `AttentionLayer` (QKV → RoPE → attention)
- `norm.py` — `RMSNorm`
- `rotary.py` — RoPE (Rotary Positional Embedding)
- `embedding.py` — `VocabParallelEmbedding`
- `activation.py` — `silu_and_mul`, `gelu_and_mul`
- `moe.py` — `MoELayer`

---

## `base.py` — 가중치 관리 프레임워크

PyTorch nn.Module 대신 자체 경량 프레임워크를 사용합니다.

### `BaseOP`
```python
def forward(self, *args) -> Any: ...     # 구현 필요

def state_dict(prefix, result) -> dict:
    # self.__dict__ 재귀 탐색 (Tensor → 저장, BaseOP → 재귀)

def load_state_dict(state_dict, prefix):
    # state_dict.pop()으로 텐서를 소비하며 로드
    # 모든 키를 소비해야 함 (남으면 RuntimeError)
```

이름이 `_`로 시작하는 속성은 무시됩니다 (내부 상태 또는 비가중치 필드).

### `StateLessOP`
가중치가 없는 레이어 (Attention, Activation 등). `state_dict()`는 빈 dict 반환.

### `OPList`
```python
OPList([layer0, layer1, ...])
# state_dict 키: "0.weight", "1.weight", ...
```

---

## `linear.py` — TP 지원 Linear 레이어

모두 `_LinearTPImpl` 상속. 실제 연산은 `F.linear(x, weight, bias)`.

| 클래스 | 샤딩 방식 | 용도 |
|--------|-----------|------|
| `LinearReplicated` | 전체 복제 | 단일 GPU와 동일 가중치 |
| `LinearColParallelMerged` | output dim 분할 | `gate_proj + up_proj` fused |
| `LinearQKVMerged` | QO/KV head 수 기준 분할 | `q_proj + k_proj + v_proj` fused |
| `LinearOProj` | input dim 분할 + all_reduce | attention output projection |
| `LinearRowParallel` | input dim 분할 + all_reduce | `down_proj` |

**all_reduce가 있는 레이어** (`LinearOProj`, `LinearRowParallel`):
```python
y = F.linear(x, weight, bias)
if tp_size > 1:
    y = communicator.all_reduce(y)   # 결과 합산
return y
```

---

## `attention.py` — `AttentionLayer`

QKV 프로젝션 결과를 받아 RoPE 적용 후 어텐션 백엔드로 위임합니다.

```python
def forward(self, qkv: Tensor) -> Tensor:
    ctx = get_global_ctx()
    q, k, v = qkv.split([qo_dim, kv_dim, kv_dim], dim=-1)

    if q_norm: q_norm.forward_inplace(q.view(-1, num_qo_heads, head_dim))
    if k_norm: k_norm.forward_inplace(k.view(-1, num_kv_heads, head_dim))

    q, k = rotary.forward(ctx.batch.positions, q, k)   # RoPE 적용

    q = q.view(-1, num_qo_heads, head_dim)
    o = ctx.attn_backend.forward(q, k, v, layer_id, ctx.batch)
    return o.view(-1, qo_attn_dim)
```

- `q_norm` / `k_norm`: Qwen3에서 사용하는 per-head RMSNorm
- TP: `num_qo_heads`, `num_kv_heads` → 로컬 head 수로 분할
- `layer_id`: 어텐션 백엔드가 올바른 KV 캐시 레이어를 참조하는 데 사용

---

## `norm.py` — `RMSNorm`

```
output = x / rms(x) * weight
rms(x) = sqrt(mean(x^2) + eps)
```

`sgl_kernel`의 CUDA 커널을 사용 (가능한 경우):
- `forward(x)` → 정규화된 텐서 반환
- `forward_inplace(x)` → 입력 텐서를 직접 수정 (q_norm, k_norm 사용)

---

## `rotary.py` — RoPE

다양한 RoPE 변형을 지원합니다.

```python
get_rope(head_dim, rotary_dim, max_position, base, rope_scaling)
```

- `rope_scaling=None`: 표준 RoPE
- `"type": "linear"`: 선형 스케일링
- `"type": "yarn"`: YaRN (Qwen3 등에서 사용)
- 디바이스 캐시: `set_rope_device(device)`로 cos/sin 테이블을 GPU에 사전 할당

`forward(positions, q, k)`:
- `positions`: `[seq_len]` int32 텐서
- `q, k`: shape `[seq_len, num_heads * head_dim]`
- 반환: RoPE 적용된 q, k (같은 shape)

---

## `embedding.py` — `VocabParallelEmbedding`

vocab 차원을 TP 랭크 수로 분할합니다.

```
전체 vocab_size = V
각 랭크: V/tp_size 크기의 임베딩 슬라이스 보유
vocab_start = rank * (V/tp_size)
vocab_end = (rank+1) * (V/tp_size)
```

`forward(input_ids)`:
1. 범위 밖 토큰은 0으로 마스크
2. 로컬 임베딩 룩업
3. `all_reduce`로 합산 (마스크된 위치는 0이므로 자동으로 올바른 값)

---

## `moe.py` — `MoELayer`

```python
def forward(self, hidden_states):
    gating_output = gate(hidden_states)   # 게이팅 스코어
    return ctx.moe_backend.forward(
        hidden_states, w1, w2, gating_output,
        topk, renormalize, activation, ...
    )
```

`w1`: `[num_experts, intermediate_size*2, hidden_size]` — gate+up fused  
`w2`: `[num_experts, hidden_size, intermediate_size]` — down

TP 지원: `w1`은 column parallel (intermediate_size 분할), `w2`는 row parallel + all_reduce.

---

## 레이어 조합 예시 (LLaMA 디코더 블록)

```
input_norm (RMSNorm)
    ↓
qkv_proj (LinearQKVMerged)  →  AttentionLayer  →  o_proj (LinearOProj)
    ↓                                [all_reduce 포함]
residual add
    ↓
ffn_norm (RMSNorm)
    ↓
gate_up_proj (LinearColParallelMerged)  →  silu_and_mul  →  down_proj (LinearRowParallel)
    ↓                                                           [all_reduce 포함]
residual add
```
