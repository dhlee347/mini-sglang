# minisgl.distributed — 분산 통신

## 역할

Tensor Parallelism을 위한 all-reduce / all-gather 통신 레이어를 제공합니다. PyTorch NCCL과 자체 PyNCCL 두 가지 구현을 플러그인 방식으로 지원합니다.

**파일**:
- `info.py` — `DistributedInfo` (TP 랭크/크기 전역 싱글턴)
- `impl.py` — `DistributedCommunicator` 및 백엔드 구현

---

## `info.py` — `DistributedInfo`

```python
@dataclass(frozen=True)
class DistributedInfo:
    rank: int   # 0 ~ size-1
    size: int   # TP 크기 (GPU 수)

    def is_primary(self) -> bool:
        return self.rank == 0
```

**전역 싱글턴 접근**:
```python
set_tp_info(rank, size)     # Engine.__init__ 시 한 번만 호출
get_tp_info() -> DistributedInfo
try_get_tp_info() -> DistributedInfo | None  # 미설정이면 None
```

---

## `impl.py` — 통신 구현

### `DistributedCommunicator`

플러그인 리스트로 백엔드를 교체 가능한 파사드:

```python
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x) -> Tensor:
        return self.plugins[-1].all_reduce(x)   # 마지막 플러그인 사용
    def all_gather(self, x) -> Tensor:
        return self.plugins[-1].all_gather(x)
```

레이어(`LinearOProj`, `LinearRowParallel`, `VocabParallelEmbedding`)에서 직접 사용:
```python
self._comm = DistributedCommunicator()
y = F.linear(x, weight)
if tp_size > 1:
    y = self._comm.all_reduce(y)
```

### `TorchDistributedImpl` (기본)
`torch.distributed.all_reduce` / `all_gather_into_tensor` 사용.
- TP=1이면 no-op 반환.

### `PyNCCLDistributedImpl`
커스텀 `PyNCCLCommunicator`를 사용하는 고성능 구현.
- `all_reduce(x, "sum")`: in-place sum all-reduce
- `all_gather(result, x)`: output_shape[0] = world_size × input_shape[0]

### `enable_pynccl_distributed(tp_info, cpu_group, max_bytes)`
```python
comm = init_pynccl(tp_rank, tp_size, cpu_group, max_bytes)
DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))
```
plugins 리스트에 PyNCCL 구현을 추가해 이후 통신이 PyNCCL을 사용하게 합니다.

Engine에서 호출 시 `max_bytes = max_forward_len * hidden_size * dtype_bytes`.

### `destroy_distributed()`
```python
DistributedCommunicator.plugins = []
```
엔진 종료 시 CUDA Graph가 먼저 소멸된 후 호출해야 합니다 (역순 해제 필요).

---

## PyNCCL vs torch.distributed 선택

```
use_pynccl=True (기본):
    torch.distributed → gloo (CPU 제어 메시지)
    PyNCCL → NCCL (GPU 텐서 통신, 더 낮은 레이턴시)

use_pynccl=False:
    torch.distributed → nccl (GPU 텐서 통신)
    + 별도 gloo group (CPU 동기화)
```

PyNCCL은 `kernel/pynccl.py`의 Python 래퍼를 통해 커스텀 CUDA NCCL 커널을 직접 호출합니다. 공유 메모리 버퍼(`max_bytes`)를 사전 할당해 레이턴시를 줄입니다.

---

## TP 동작 흐름 (TP=4 예시)

```
Scheduler Rank 0           Scheduler Rank 1,2,3
    ↓                            ↓
ZMQ 수신 (UserMsg)        ZMQ sub (브로드캐스트 수신)
    ↓                            ↓
         같은 배치 처리 (동기화됨)
    ↓                            ↓
Engine.forward_batch()   Engine.forward_batch()
    ↓                            ↓
LinearQKV → all_reduce   LinearQKV → all_reduce
    ↓                            ↓ (NCCL 동기 실행)
AttentionLayer           AttentionLayer
(로컬 head 담당)         (로컬 head 담당)
    ↓                            ↓
LinearOProj → all_reduce LinearOProj → all_reduce
    ↓                            ↓ (NCCL 동기 실행)
    ...
    ↓
Rank 0만 logits 전송 → Sampler → DetokenizeMsg
```
