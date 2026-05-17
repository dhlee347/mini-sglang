# minisgl.engine — 단일 GPU 추론 엔진

## 역할

하나의 GPU에서 실제 모델 포워드 패스를 실행하는 워커입니다. 모델, KV 캐시, 어텐션 백엔드, CUDA Graph, 샘플러를 통합 관리합니다.

**파일**:
- `engine.py` — `Engine` 클래스
- `config.py` — `EngineConfig` dataclass
- `graph.py` — `GraphRunner` (CUDA Graph 캡처 및 재실행)
- `sample.py` — `Sampler` (greedy / top-k / top-p)

---

## `config.py` — `EngineConfig`

```python
model_path: str
tp_info: DistributedInfo
dtype: torch.dtype
max_running_req: int = 256
attention_backend: str = "auto"   # "fa", "fi", "trtllm", "fa,fi", "auto"
moe_backend: str = "auto"
cuda_graph_bs: List[int] | None = None
cuda_graph_max_bs: int | None = None
page_size: int = 1
memory_ratio: float = 0.9         # KV 캐시에 쓸 GPU 메모리 비율
use_dummy_weight: bool = False
use_pynccl: bool = True
```

**lazy 프로퍼티**:
- `hf_config`: HuggingFace config (처음 접근 시 로드 + 캐시)
- `model_config`: `ModelConfig` (from HF config)
- `max_seq_len`: rotary position 최대값 (또는 override)

---

## `engine.py` — `Engine`

### 초기화 순서 (`__init__`)

```
1. TP 정보 설정, CUDA 디바이스 초기화
2. 분산 통신 그룹 초기화 (_init_communication)
3. 모델 초기화 (meta device → 가중치 로드)
4. KV 캐시 페이지 수 결정 (_determine_num_pages)
5. MHAKVCache 할당
6. page_table 초기화 (max_running_req × aligned_max_seq_len)
7. 어텐션 백엔드 생성 (_adjust_config → auto 선택)
8. MoE 백엔드 생성 (MoE 모델인 경우)
9. Sampler 초기화
10. GraphRunner 초기화 (CUDA Graph 캡처)
```

### `_adjust_config()` — 자동 백엔드 선택

```python
if attention == "auto":
    sm100 (Blackwell) → "trtllm"
    sm90  (Hopper)    → "fa,fi"  # prefill=FA, decode=FI
    기타              → "fi"

if "trtllm" in backend and page_size not in [16,32,64]:
    page_size = 64  # TensorRT-LLM 제약

if is_moe and moe_backend == "auto":
    moe_backend = "fused"
```

### `_init_communication()` — 분산 통신 설정

```
TP=1 또는 use_pynccl=True:
    → torch.distributed (gloo backend)
    → enable_pynccl_distributed() (NCCL 오버레이)

use_pynccl=False (멀티GPU):
    → torch.distributed (nccl backend)
    → 별도 gloo group (CPU 동기화용)
```

### `_determine_num_pages()` — KV 캐시 크기 결정

```
model_memory = init_free_mem - post_load_free_mem
available_memory = memory_ratio × init_free_mem - model_memory
num_pages = available_memory / (2 × head_dim × local_kv_heads × page_size × dtype_bytes × num_layers)
```

TP 랭크 간 free memory가 2GB 이상 차이나면 RuntimeError.

### `forward_batch(batch, args)` → `ForwardOutput`

```python
with ctx.forward_batch(batch):
    if graph_runner.can_use_cuda_graph(batch):
        logits = graph_runner.replay(batch)     # CUDA Graph 재실행
    else:
        logits = model.forward()                # 일반 포워드

for req in batch.reqs:
    req.complete_one()                          # cached_len, device_len 갱신

next_tokens_gpu = sampler.sample(logits[:batch.size], args)
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
copy_done_event.record(stream)                  # 비동기 복사 이벤트
```

---

## `graph.py` — `GraphRunner`

### CUDA Graph 캡처

decode 배치의 배치 크기별로 CUDA Graph를 미리 캡처합니다.

**캡처 배치 크기 자동 결정**:
```python
[1, 2, 4] + range(8, cuda_graph_max_bs+1, 8)
# H200 (>80GB): max=256
# 기타: max=160
```

**캡처 방식**:
```python
for bs in sorted(bs_list, reverse=True):  # 큰 것부터 (메모리 재사용 위해)
    attn_backend.prepare_for_capture(batch)  # 캡처용 더미 메타데이터 세팅
    buffer.set_batch(batch)                  # 고정 입력 버퍼 세팅
    with ctx.forward_batch(batch):
        model.forward()  # warm-up
        with torch.cuda.graph(graph, pool=pool):
            model.forward()  # 캡처
    pool = graph.pool()  # 메모리 풀 재사용
    graph_map[bs] = graph
```

### `GraphCaptureBuffer`
캡처된 그래프가 참조하는 고정 입력/출력 텐서:
```python
input_ids: Tensor    # [max_bs]
out_loc: Tensor      # [max_bs]
positions: Tensor    # [max_bs]
logits: Tensor       # [max_bs, vocab_size]
```

### `replay(batch)` — 그래프 재실행
1. `buffer.copy_from(batch)` — 실제 입력을 버퍼에 복사
2. `attn_backend.prepare_for_replay(batch)` — 어텐션 메타데이터 갱신
3. `graph.replay()` — GPU에서 캡처된 커널 재실행
4. `buffer.logits[:batch.size]` 반환

### `pad_batch(batch)` — CUDA Graph 패딩
decode 배치의 `padded_reqs`를 `graph_bs_list`에서 `>= batch.size`인 최소 크기로 dummy req 패딩.

---

## `sample.py` — `Sampler`

### `BatchSamplingArgs`
```python
temperatures: Tensor | None   # None이면 greedy (argmax)
top_k: Tensor | None
top_p: Tensor | None
```

### `Sampler.prepare(batch)` → `BatchSamplingArgs`

모든 요청이 greedy이면 `temperatures=None` 반환 (가장 빠른 경로).

그렇지 않으면:
- `temperature`: `is_greedy`인 요청은 MIN_T (1e-6)으로 처리
- `top_k`: `vocab_size`와 같으면 None (비활성)
- `top_p`: 1.0이면 None (비활성)

### `Sampler.sample(logits, args)` → `Tensor`

```python
if args.temperatures is None:
    return torch.argmax(logits, dim=-1)   # greedy

probs = flashinfer.sampling.softmax(logits, temperatures)

if top_k and top_p:
    return top_k_top_p_sampling_from_probs(probs, top_k, top_p)
elif top_k:
    return top_k_sampling_from_probs(probs, top_k)
elif top_p:
    return top_p_sampling_from_probs(probs, top_p)
else:
    return sampling_from_probs(probs)
```

FlashInfer의 GPU 가속 샘플링 커널 사용. SM90 이상에서는 PDL(Persistent Device Launch) 활성화.
