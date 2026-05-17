# minisgl.moe — Mixture of Experts 백엔드

## 역할

MoE(Mixture of Experts) 레이어의 게이팅 및 전문가 연산을 최적화된 방식으로 실행합니다.

**파일**:
- `base.py` — `BaseMoeBackend` 추상 클래스
- `fused.py` — `FusedMoe` (Triton + sgl_kernel 기반 구현)

---

## `base.py` — `BaseMoeBackend`

```python
class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: Tensor,    # [num_tokens, hidden_size]
        w1: Tensor,               # [num_experts, intermediate_size*2, hidden_size]
        w2: Tensor,               # [num_experts, hidden_size, intermediate_size]
        gating_output: Tensor,    # [num_tokens, num_experts] — 라우터 스코어
        topk: int,
        renormalize: bool,
        activation: str,          # "silu" 또는 "gelu"
        apply_router_weight_on_input: bool,
    ) -> Tensor: ...
```

---

## `fused.py` — `FusedMoe`

Triton 커널과 sgl_kernel을 조합한 고성능 구현.

### 처리 흐름

```
1. fused_topk()          → topk_weights [M, topk], topk_ids [M, topk]
2. moe_align_block_size() → sorted_token_ids, expert_ids, num_tokens_padded
3. fused_moe_kernel_triton(hidden, w1, ...) → intermediate_cache1 [M, topk, N]
4. activation(intermediate_cache1)          → intermediate_cache2 [M*topk, N//2]
5. fused_moe_kernel_triton(cache2, w2, ...) → intermediate_cache3 [M, topk, hidden]
6. moe_sum_reduce_triton(cache3)            → out_hidden_states [M, hidden]
```

### `fused_topk(hidden, gating, topk, renormalize)`

`sgl_kernel.topk_softmax`를 사용해 게이팅 스코어에서 top-k 전문가를 선택합니다.

```python
topk_weights: [M, topk]   # 각 토큰의 top-k 전문가 가중치 (softmax 정규화)
topk_ids: [M, topk]        # 각 토큰이 라우팅될 전문가 인덱스
```

`renormalize=True`: `topk_weights / sum(topk_weights)` 재정규화.

### `moe_align_block_size(topk_ids, block_size, num_experts)`

Triton 커널의 블록 행렬 연산을 위해 토큰을 전문가별로 정렬하고 패딩합니다.

```
topk_ids = [[2,3], [1,2], [1,3]] (3 tokens, topk=2)
→ 전문가 1: tokens [1,2]
→ 전문가 2: tokens [0,1]
→ 전문가 3: tokens [0,2]
→ block_size=4로 패딩 후 정렬된 sorted_token_ids 반환
```

반환값:
- `sorted_token_ids`: 전문가별로 정렬된 토큰 인덱스 (패딩 포함)
- `expert_ids`: 각 블록이 담당하는 전문가 인덱스
- `num_tokens_post_padded`: 패딩 후 총 토큰 수

### `fused_moe_kernel_triton`

`minisgl.kernel.triton.fused_moe`의 Triton 커널 호출. 블록 행렬 방식으로 MoE 연산을 수행합니다.

**튜닝 파라미터** (`get_default_config`):
```python
# 기본값
BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8

# M <= num_experts (소규모 배치)
BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1
```

### 메모리 최적화

`cache` 텐서를 재사용해 피크 메모리를 줄입니다:
```python
cache = torch.empty(M * topk * max(N, hidden_size), ...)
intermediate_cache1 = cache[:M*topk*N].view(M, topk, N)
intermediate_cache3 = cache[:M*topk*hidden].view(M, topk, hidden)
# cache1과 cache3이 같은 버퍼를 공유 (순차 실행이므로 안전)
```

---

## TP에서의 MoE

`MoELayer`에서 TP를 적용할 때:

```
w1 [num_experts, N*2, hidden] → N 차원을 tp_size로 분할 (column parallel)
w2 [num_experts, hidden, N]   → N 차원을 tp_size로 분할 + all_reduce
```

각 랭크는 전체 expert 중 자신의 몫을 처리하고 결과를 합산합니다.

---

## 백엔드 자동 선택

```python
if model_config.is_moe and moe_backend == "auto":
    moe_backend = "fused"
```

현재 `fused` 백엔드만 지원.
