# minisgl.attention — 어텐션 백엔드

## 역할

Multi-Head Attention 연산의 백엔드 인터페이스를 정의하고, FlashAttention, FlashInfer, TensorRT-LLM FMHA 세 가지 구현을 제공합니다. 동일 인터페이스를 통해 prefill/decode별로 다른 백엔드를 사용할 수 있습니다.

**파일**:
- `base.py` — 추상 베이스 클래스 및 `HybridBackend`
- `fa.py` — FlashAttention 백엔드
- `fi.py` — FlashInfer 백엔드
- `trtllm.py` — TensorRT-LLM FMHA 백엔드
- `utils.py` — CUDA Graph 캡처용 공통 버퍼 (`BaseCaptureData`)

---

## `base.py` — 추상 인터페이스

### `BaseAttnMetadata`
어텐션 커널에 전달할 메타데이터의 추상 클래스.

```python
@abstractmethod
def get_last_indices(self, bs: int) -> torch.Tensor:
    # 배치의 각 요청에서 마지막 토큰의 logit 위치를 반환
    # Sampler가 decode 토큰 logit을 추출할 때 사용
```

### `BaseAttnBackend`
모든 어텐션 백엔드가 구현해야 하는 인터페이스:

| 메서드 | 호출 시점 | 설명 |
|--------|-----------|------|
| `forward(q, k, v, layer_id, batch)` | 레이어 포워드 | KV 저장 + attention 연산 |
| `prepare_metadata(batch)` | 배치 실행 직전 | `batch.attn_metadata` 세팅 |
| `init_capture_graph(max_seq_len, bs_list)` | 엔진 초기화 | 캡처용 버퍼/wrapper 준비 |
| `prepare_for_capture(batch)` | 각 bs 캡처 직전 | 더미 메타데이터로 메모리 고정 |
| `prepare_for_replay(batch)` | 그래프 재실행 직전 | 실제 메타데이터를 버퍼에 복사 |

### `HybridBackend`
prefill/decode에 서로 다른 백엔드를 사용할 때의 라우터.

```python
HybridBackend(prefill_backend=FA, decode_backend=FI)
# batch.is_prefill → FA.forward()
# batch.is_decode  → FI.forward()
# CUDA graph 캡처/재실행은 decode_backend에 위임
```

`--attn fa,fi` 형식으로 지정 시 자동 생성됨.

---

## `fa.py` — `FlashAttentionBackend`

`sgl_kernel.flash_attn.flash_attn_with_kvcache`를 사용. Hopper(SM90)는 FA3, Blackwell(SM100)은 FA4 버전 선택.

### `FAMetadata`
```python
cu_seqlens_k: Tensor    # K 시퀀스 누적 길이 [bs+1]
cu_seqlens_q: Tensor    # Q 시퀀스 누적 길이 [bs+1]
cache_seqlens: Tensor   # 각 요청의 K 길이 [bs]
max_seqlen_k: int
max_seqlen_q: int
page_table: Tensor      # [bs, max_k // page_size] — FA 페이지 인덱스
```

### `prepare_metadata(batch)` — 세 가지 경우

| 경우 | cu_seqlens_q |
|------|-------------|
| decode (max_q=1) | `[0, 1, 2, ..., bs]` |
| prefill + no cache hit | `cu_seqlens_k`와 동일 |
| prefill + partial cache | 별도 계산 ([0] + seqlens_q 누적합) |

**page_table 변환**: 전역 page_table(page_size=1 기준)을 FA가 요구하는 페이지 단위 인덱스로 변환. `new_page_table = page_table[req.table_idx, ::page_size] // page_size`

### `forward(q, k, v, layer_id, batch)`
1. `kvcache.store_kv(k, v, batch.out_loc, layer_id)` — KV 캐시에 저장
2. `_fa_sgl_impl(...)` — FA 어텐션 연산

---

## `fi.py` — `FlashInferBackend`

`flashinfer` 라이브러리의 paged KV cache wrapper를 사용.

**현재 제약**: `page_size=1`만 지원 (indices가 flat 토큰 인덱스로 처리됨).

### `FIMetadata`
```python
cu_seqlens_q_cpu / gpu: Tensor
cu_seqlens_k_cpu: Tensor
indices: Tensor            # 플랫한 토큰 인덱스 (모든 req 연결)
last_page_len_cpu: Tensor  # 항상 [1, 1, ..., 1] (page_size=1)
wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
initialized: bool          # 한 번만 plan() 호출하도록 지연 초기화
```

### 지연 초기화 (`_initialize_metadata_once`)
FlashInfer의 `plan()` 호출은 pinned host 버퍼를 사용해 async H2D 복사를 트리거합니다. 이전 plan이 완료될 때까지 `last_event.synchronize()`로 대기합니다.

### Tensor Core 사용 (`use_tensor_cores`)
```python
GQA (num_qo_heads // num_kv_heads) >= 4 → tensor core 사용
MINISGL_FLASHINFER_USE_TENSOR_CORES 환경변수로 override 가능
```

### CUDA Graph wrapper (`CUDAGraphBatchDecodeWithPagedKVCacheWrapper`)
bs별로 별도 wrapper 생성. `graph_wrappers: Dict[int, wrapper]`에 캐싱. `prepare_for_capture` 시 생성, `prepare_for_replay` 시 `metadata.wrapper` 교체.

---

## `utils.py` — `BaseCaptureData`

CUDA Graph 캡처를 위한 고정 크기 버퍼. 캡처 후 실제 값을 `copy_`로 덮어씁니다.

```python
seq_lens: Tensor     # [max_bs]
positions: Tensor    # [max_bs]
cu_seqlens_k: Tensor # [max_bs+1]
cu_seqlens_q: Tensor # [max_bs+1]
page_table: Tensor   # [max_bs, max_seq_len]
```

---

## 백엔드 선택 기준 요약

| GPU | 자동 선택 | 특징 |
|-----|-----------|------|
| Blackwell (SM100) | `trtllm` | page_size=16/32/64 제약 |
| Hopper (SM90) | `fa,fi` | prefill=FA3, decode=FI |
| 기타 | `fi` | FlashInfer 단독 |

---

## 다른 모듈과의 관계

```
Engine.__init__
    → create_attention_backend(config.attention_backend)
    → Context.attn_backend = backend

AttentionLayer.forward(qkv)
    → ctx.attn_backend.forward(q, k, v, layer_id, batch)
    → kvcache.store_kv() 내부 호출

Scheduler._prepare_batch(batch)
    → attn_backend.prepare_metadata(batch)

GraphRunner._capture_graphs()
    → attn_backend.init_capture_graph()
    → attn_backend.prepare_for_capture()
    → attn_backend.prepare_for_replay()
```
