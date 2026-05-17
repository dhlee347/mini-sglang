# minisgl.scheduler — 스케줄러

## 역할

각 GPU(TP 랭크)마다 하나씩 실행되는 핵심 워커 프로세스입니다. 요청 큐를 관리하고, prefill/decode 배치를 구성하여 Engine에 실행을 위임하며, 결과를 디토크나이저로 전달합니다. Overlap Scheduling을 통해 CPU 스케줄링과 GPU 연산을 겹칩니다.

**파일**:
- `scheduler.py` — 메인 스케줄러 루프 및 Overlap Scheduling
- `prefill.py` — Chunked Prefill 및 prefill 배치 구성
- `decode.py` — decode 배치 구성
- `cache.py` — KV 캐시 페이지 할당 및 prefix cache 연동
- `table.py` — page_table 슬롯(row) 관리
- `io.py` — ZMQ 송수신 mixin (단일/멀티 랭크 분기)
- `config.py` — `SchedulerConfig` dataclass
- `utils.py` — `PendingReq` 등 내부 유틸

---

## `config.py` — `SchedulerConfig`

`EngineConfig`를 상속하는 frozen dataclass.

```python
max_extend_tokens: int = 8192   # Chunked Prefill 최대 청크 크기
cache_type: str = "radix"       # "radix" 또는 "naive"
offline_mode: bool = False      # LLM 클래스(오프라인)에서 사용 시 True
_unique_suffix: str             # PID 기반 ZMQ 주소 suffix
```

ZMQ 주소 프로퍼티: `zmq_backend_addr`, `zmq_detokenizer_addr`, `zmq_scheduler_broadcast_addr`

---

## `scheduler.py` — 메인 루프

### `Scheduler.__init__`
Engine을 생성하고 다음 매니저들을 초기화:
- `TableManager`: page_table 슬롯 할당
- `CacheManager`: KV 캐시 페이지 + prefix cache
- `DecodeManager`: 실행 중인 decode 요청 집합
- `PrefillManager`: pending 요청 큐 및 prefill 배치 구성

### 실행 루프

#### `run_forever()` — 두 가지 모드

**Normal mode** (`MINISGL_DISABLE_OVERLAP_SCHEDULING=1`):
```
메시지 수신 → 배치 구성 → 포워드 → 결과 처리 → 반복
```

**Overlap mode** (기본):
```
overlap_loop(last_data=None)
    ├─ 메시지 수신 (non-blocking if last_data 있음)
    ├─ 배치 구성 (_schedule_next_batch)
    ├─ 포워드 실행 (engine stream에서 비동기 실행) ─┐
    ├─ 이전 배치 결과 처리 (_process_last_data)    │
    └─ 다음 루프에 현재 배치 결과 전달 ←───────────┘
```

CPU 스케줄링(이전 배치 처리)과 GPU 포워드(현재 배치)가 병렬 실행됩니다.

### `_schedule_next_batch()` — 스케줄링 정책
Prefill 우선: `PrefillManager.schedule_next_batch()` → (없으면) `DecodeManager.schedule_next_batch()`

### `_prepare_batch(batch)` — 배치 실행 준비
```python
graph_runner.pad_batch(batch)           # CUDA graph용 dummy req 패딩
cache_manager.allocate_paged(batch)     # 새 KV 캐시 페이지 할당
batch.positions = _make_positions(batch) # RoPE 위치 인덱스 구성
input_mapping = _make_input_tuple(batch) # token_pool 조회용 인덱스
write_mapping = _make_write_tuple(batch) # KV 쓰기 위치
batch.out_loc = page_table[input_mapping] # GPU KV 저장 위치
attn_backend.prepare_metadata(batch)     # 어텐션 메타데이터 구성
```

### `_process_last_data(last_data)` — 결과 처리
```python
copy_done.synchronize()            # GPU→CPU 복사 완료 대기
for req, next_token in batch.reqs:
    req.append_host(next_token)    # CPU input_ids에 추가
    finished = not can_decode or is_eos
    DetokenizeMsg(uid, next_token, finished) → send_result()
    if finished: free_req_resources(req)
    elif prefill: cache_req(req, finished=False)  # prefix 캐싱
```

### 텐서 구성 헬퍼 함수

| 함수 | 역할 |
|------|------|
| `_make_positions(batch)` | 각 req의 `[cached_len, device_len)` 범위로 위치 텐서 구성 |
| `_make_input_tuple(batch)` | `token_pool[table_idx, pos]` 조회용 2D 인덱스 |
| `_make_write_tuple(batch)` | `token_pool[table_idx, device_len]` 쓰기용 인덱스, 완료된 req는 -1 sentinel |

---

## `prefill.py` — Chunked Prefill

### `ChunkedReq`
`Req`를 상속하며 `append_host()`와 `can_decode`를 오버라이드해 decode 단계 진입을 막습니다. 긴 프롬프트를 여러 청크로 나눌 때 사용됩니다.

### `PrefillAdder`
배치 구성 중 개별 요청 추가를 담당합니다.

```python
token_budget: int     # 이번 prefill에서 처리 가능한 최대 토큰 수
reserved_size: int    # decode 요청들의 예상 KV 소비량
```

`try_add_one(pending_req)` 처리 흐름:
1. `table_manager.available_size` 확인
2. `cache_manager.match_req()` — prefix cache 매칭
3. 예상 KV 크기 추정 (`extend_len + output_len + reserved_size`)
4. KV 공간 부족 시 `None` 반환
5. `table_manager.allocate()` — page_table 슬롯 확보
6. 캐시된 부분의 page 인덱스를 page_table에 복사
7. `token_budget` 내 청크 크기 결정 → `ChunkedReq` 또는 `Req` 생성

### `PrefillManager`
```python
pending_list: List[PendingReq]   # 대기 중인 요청들 (FIFO)
```

`schedule_next_batch(prefill_budget)`:
- `PrefillAdder`로 순서대로 요청 추가 시도
- `ChunkedReq`로 분할된 요청은 다음 라운드에서 이어서 처리
- 추가 실패 시 break (이후 요청도 자원 부족으로 실패할 가능성 높음)

---

## `decode.py` — `DecodeManager`

```python
running_reqs: Set[Req]    # 현재 decode 중인 요청들
```

- `filter_reqs(reqs)`: 포워드 후 `can_decode=False` 요청 제거
- `schedule_next_batch()`: `running_reqs` 전체를 decode 배치로 반환
- `inflight_tokens`: 현재 decode 요청들의 예상 KV 소비량 (reserved_size 계산에 사용)

---

## `cache.py` — `CacheManager`

KV 캐시 페이지 풀과 prefix cache를 함께 관리합니다.

```python
free_slots: torch.Tensor    # 사용 가능한 페이지 시작 위치들 (page-aligned)
prefix_cache: BasePrefixCache  # RadixPrefixCache 또는 NaivePrefixCache
```

**핵심 메서드**:

| 메서드 | 설명 |
|--------|------|
| `allocate_paged(reqs)` | 요청들이 새로 필요한 페이지를 free_slots에서 할당 → page_table 기록 |
| `cache_req(req, finished)` | 요청 완료/prefill 후 prefix cache에 삽입, 불필요 페이지 반환 |
| `match_req(pending)` | prefix cache에서 입력 prefix 매칭 (`match_prefix`) |
| `lock(handle)` / `unlock(handle)` | 매칭된 캐시 항목을 eviction으로부터 보호 |
| `lazy_free_region()` | context manager — 해제할 페이지를 즉시 반환하지 않고 배치 처리 (오버랩 스케줄링 안전) |

`cache_req`의 메모리 관리:
```
[0, old_handle.cached_len)           → prefix cache에 이미 존재, 건드리지 않음
[old_handle.cached_len, cached_len)  → 다른 요청에 의해 이미 캐시됨 → 즉시 반환
[cached_len, new_handle.cached_len)  → 새로 insert됨 → prefix cache가 소유
[new_handle.cached_len, req.cached_len) → 꼬리 부분
  └─ finished=True  → free_slots 반환
  └─ finished=False → 계속 사용 (handle 갱신)
```

---

## `table.py` — `TableManager`

page_table의 행(슬롯) 단위 할당을 관리합니다.

```python
_free_slots: List[int]   # 사용 가능한 table_idx 목록
token_pool: Tensor       # page_table과 같은 크기의 int32 텐서 (토큰 ID 저장)
```

- `allocate()`: free_slots에서 pop — O(1)
- `free(slot)`: free_slots에 append — O(1)

`token_pool`은 GPU 상의 토큰 ID 버퍼로, 스케줄러가 `_make_input_tuple` / `_forward`에서 참조합니다.

---

## `io.py` — `SchedulerIOMixin`

TP 랭크 수에 따라 ZMQ 통신 방식을 자동 선택합니다.

| 상황 | `receive_msg` | `send_result` |
|------|---------------|---------------|
| TP=1 | PULL from tokenizer | PUSH to detokenizer |
| TP>1 Rank 0 | PULL + PUB 브로드캐스트 | PUSH to detokenizer |
| TP>1 Rank N | SUB from Rank 0 | 아무것도 안 함 |

멀티랭크 동기화: `tp_cpu_group.broadcast(msg_count)` — Rank 0가 메시지 수를 다른 랭크에 브로드캐스트해서 처리할 메시지 수를 맞춤.
