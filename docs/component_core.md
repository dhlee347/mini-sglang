# minisgl.core — 핵심 데이터클래스

## 역할

시스템 전체에서 공유하는 핵심 데이터 구조를 정의합니다. 요청(Req), 배치(Batch), 전역 추론 컨텍스트(Context), 샘플링 파라미터(SamplingParams)가 이 모듈에서 관리됩니다.

**파일**: `python/minisgl/core.py`

---

## 클래스별 설명

### `SamplingParams`
사용자가 요청 시 지정하는 샘플링 파라미터를 담는 dataclass.

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `temperature` | 0.0 | 샘플링 온도 (0이면 greedy) |
| `top_k` | -1 | top-k 필터 (-1이면 비활성) |
| `top_p` | 1.0 | top-p 누클리어스 샘플링 |
| `ignore_eos` | False | EOS 토큰 무시 여부 |
| `max_tokens` | 1024 | 최대 생성 토큰 수 |

`is_greedy` 프로퍼티: temperature ≤ 0 또는 top_k == 1이고 top_p == 1.0일 때 True.

---

### `Req`
단일 요청의 상태를 추적하는 dataclass. 스케줄러와 엔진이 공유하며, 요청 생애주기 전체에 걸쳐 변경됩니다.

```
input_ids: torch.Tensor   # CPU 1D int32 텐서 (현재까지 생성된 모든 토큰 포함)
table_idx: int            # page_table에서 이 요청이 점유하는 행 인덱스
cached_len: int           # prefix cache에 이미 저장된 토큰 수 (attention 불필요)
device_len: int           # GPU에 올라온 토큰 수 (prefill 완료된 부분)
max_device_len: int       # input_len + max_output_len
uid: int                  # 요청 고유 ID
sampling_params: SamplingParams
cache_handle: BaseCacheHandle  # kvcache 매니저의 핸들
```

**핵심 프로퍼티**:
- `extend_len = device_len - cached_len` — 이번 포워드 패스에서 새로 처리할 토큰 수
- `remain_len = max_device_len - device_len` — 아직 생성하지 않은 토큰 수
- `can_decode` — `remain_len > 0`이면 True (아직 생성할 토큰 있음)

**핵심 메서드**:
- `complete_one()`: 토큰 하나 생성 후 `cached_len`, `device_len` 갱신
- `append_host(next_token)`: 생성된 토큰을 `input_ids` CPU 텐서에 이어붙임

---

### `Batch`
스케줄러가 엔진에게 넘겨주는 미니배치. prefill과 decode 두 phase 중 하나입니다.

```
reqs: List[Req]            # 실제 요청 목록
phase: "prefill" | "decode"
input_ids: torch.Tensor   # GPU 텐서 (스케줄러가 세팅)
positions: torch.Tensor   # GPU 텐서 — RoPE 계산용 위치 인덱스
out_loc: torch.Tensor     # GPU 텐서 — KV 저장 위치 (page_table 기반)
padded_reqs: List[Req]    # CUDA graph를 위해 dummy req로 패딩된 목록
attn_metadata: BaseAttnMetadata  # attention 백엔드가 세팅
```

---

### `Context`
단일 GPU 추론 엔진(Engine)의 전역 상태. `get_global_ctx()`로 어디서든 접근 가능합니다.

```
page_size: int
page_table: torch.Tensor    # [max_req, max_seq_len] int32 GPU 텐서
                             # page_size=1 기준으로 저장 (raw token 위치)
attn_backend: BaseAttnBackend
moe_backend: BaseMoeBackend
kv_cache: BaseKVCachePool
_batch: Batch | None        # forward_batch() context 내에서만 비어있지 않음
```

`forward_batch(batch)` context manager: 포워드 패스 동안 현재 배치를 context에 등록하고, 레이어들이 `ctx.batch`로 접근할 수 있게 합니다.

---

## 전역 컨텍스트 접근

```python
set_global_ctx(ctx)    # Engine 초기화 시 한 번만 호출
get_global_ctx()       # 레이어, 어텐션 백엔드 등에서 사용
```

프로세스당 하나의 Context만 허용되며, 중복 설정 시 AssertionError.

---

## 다른 모듈과의 관계

```
SamplingParams ──→ UserMsg (message) ──→ Scheduler.add_one_req
Req            ──→ Batch ──→ Engine.forward_batch
Context        ──→ AttentionLayer.forward (get_global_ctx())
Context        ──→ attention 백엔드 (prepare_metadata, forward)
```
