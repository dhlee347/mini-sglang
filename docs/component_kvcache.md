# minisgl.kvcache — KV 캐시 시스템

## 역할

GPU 메모리 상의 Key-Value 캐시를 관리하는 두 계층의 추상화를 제공합니다.
- **KVCachePool**: GPU 텐서를 직접 관리 (저장/조회)
- **PrefixCache**: 공유 prefix의 재사용 가능 여부를 추적 (RadixTree 또는 Naive)

**파일**:
- `base.py` — 추상 베이스 클래스 및 공통 타입
- `mha_pool.py` — `MHAKVCache` (실제 GPU 텐서 풀)
- `radix_cache.py` — `RadixPrefixCache` (Radix Tree 기반 prefix 재사용)
- `naive_cache.py` — `NaivePrefixCache` (prefix 재사용 없음)

---

## `base.py` — 추상 인터페이스

### `BaseKVCachePool`
GPU 텐서에 KV를 저장하고 꺼내는 인터페이스:

```python
k_cache(index: int) -> Tensor   # 레이어 index의 K 텐서
v_cache(index: int) -> Tensor   # 레이어 index의 V 텐서
store_kv(k, v, out_loc, layer_id)  # 토큰 위치(out_loc)에 K, V 저장
```

### `BaseCacheHandle`
prefix cache 매칭 결과 핸들:
```python
cached_len: int                    # 이 핸들이 커버하는 토큰 수
get_matched_indices() -> Tensor    # page_table에 복사할 캐시 위치 인덱스
```

### `BasePrefixCache`
prefix cache 관리 인터페이스:

| 메서드 | 설명 |
|--------|------|
| `match_prefix(input_ids)` | prefix 매칭 → `MatchResult(cuda_handle)` |
| `insert_prefix(input_ids, indices)` | prefix 삽입 → `InsertResult(cached_len, handle)` |
| `lock_handle(handle, unlock)` | eviction 보호 (lock)/해제 (unlock) |
| `evict(size)` | `size` 토큰만큼 강제 evict → 반환된 인덱스 |
| `size_info` | `SizeInfo(evictable_size, protected_size)` |
| `check_integrity()` | 상태 무결성 검증 |

### 공통 반환 타입
```python
MatchResult(cuda_handle: BaseCacheHandle)
InsertResult(cached_len: int, handle: BaseCacheHandle)
SizeInfo(evictable_size, protected_size)
```

---

## `mha_pool.py` — `MHAKVCache`

Multi-Head Attention용 KV 캐시 GPU 텐서 풀입니다.

### 텐서 구조
```python
_kv_buffer: Tensor  # shape: (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
                     # [0]: K, [1]: V
```

TP를 고려해 `local_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)`.

### `store_kv(k, v, out_loc, layer_id)`
`minisgl.kernel.store_cache` CUDA 커널을 호출합니다.
- `k, v`: 새로 계산된 현재 배치의 Key, Value 텐서
- `out_loc`: 각 토큰이 저장될 GPU 내 위치 (`batch.out_loc`, page_table 기반)
- `layer_id`: 저장할 레이어 인덱스

`store_cache`는 `out_loc[i]`에 해당하는 KV buffer 위치에 scatter 씁니다.

---

## `radix_cache.py` — `RadixPrefixCache`

SGLang의 Radix Attention을 구현합니다. 입력 토큰의 공유 prefix를 Radix Tree(Trie)로 관리해 KV 캐시를 재사용합니다.

### `RadixTreeNode`
트리의 노드 하나가 일정 길이의 토큰 시퀀스(key)와 그 KV 인덱스(value)를 담습니다.

```python
children: Dict[key_fn(tokens), RadixTreeNode]
_key: Tensor         # 이 노드가 담당하는 토큰 ID 시퀀스
_value: Tensor       # 대응하는 KV 캐시 인덱스
ref_count: int       # 0이면 evict 가능
timestamp: int       # LRU eviction을 위한 최근 접근 시각 (ns)
```

**`split_at(pos)`**: prefix 길이가 맞지 않을 때 노드를 분할합니다.
```
[ABCDE] → [AB] + [CDE]   (pos=2에서 분할)
```

**`key_fn`**: 노드 식별 키 함수. `page_size=1`이면 첫 토큰 값, `page_size>1`이면 첫 페이지 토큰들의 tuple.

### `RadixCacheHandle`
매칭된 노드에서 루트까지 value 텐서를 역순으로 수집해 `get_matched_indices()`를 구현합니다.

### 핵심 메서드

#### `match_prefix(input_ids)` — prefix 탐색
`_tree_walk(input_ids)`: 루트부터 시작해 key_fn으로 child 탐색.
- `fast_compare_key(node._key, input_ids)` — CUDA 커널로 빠른 매칭 길이 계산
- 완전 매칭: 다음 노드로 이동, timestamp 갱신
- 부분 매칭: `split_at(match_len)` 후 반환

반환 길이는 항상 `page_size`의 배수로 정렬.

#### `insert_prefix(input_ids, indices)` — prefix 삽입
```
insert_len = align_down(len(input_ids), page_size)  # 꼬리 토큰 제외
prefix_len = _tree_walk까지 이미 매칭된 길이
if prefix_len < insert_len:
    새 노드 생성 (input_ids[prefix_len:], indices[prefix_len:].clone())
    evictable_size 증가
```

#### `evict(size)` — LRU eviction
1. leaf 노드 중 `ref_count == 0`인 것들만 수집
2. `heapq`로 timestamp 오래된 순 정렬
3. `size` 이상 될 때까지 leaf부터 제거
4. 부모가 leaf가 되면 heap에 추가 (연쇄 eviction 가능)
5. 제거된 노드의 value 텐서 연결 후 반환

#### `lock_handle(handle, unlock)` — 보호 상태 변경
노드부터 루트 방향으로 ref_count를 +1/-1. `ref_count > 0`이면 eviction 불가.

---

## `naive_cache.py` — `NaivePrefixCache`

prefix 재사용 없이 매번 전체를 새로 계산하는 단순 구현.

- `match_prefix` → 항상 `NaiveCacheHandle(cached_len=0)` 반환
- `insert_prefix` → 아무것도 저장하지 않고 `InsertResult(0, ...)` 반환
- `evict` → size=0이면 빈 텐서, 아니면 `NotImplementedError`
- `size_info` → 항상 `SizeInfo(0, 0)`

`--cache naive` 옵션 시 사용. prefix 공유가 없는 벤치마크용.

---

## 메모리 관리 흐름

```
요청 추가 시:
  match_prefix() → 매칭된 cached_len 확인
  lock(handle)   → eviction 보호
  page 할당 (free_slots에서)

포워드 후 (prefill):
  cache_req(req, finished=False)
    → insert_prefix(input_ids, page_indices)
    → old_cached 부분 free (중복 페이지)
    → tail 부분 lock (계속 사용)

요청 완료 시:
  cache_req(req, finished=True)
    → insert_prefix()
    → tail 부분 free_slots에 반환
    → unlock(old_handle)

메모리 부족 시:
  evict(needed_pages * page_size) → evicted 인덱스 → free_slots에 추가
```
