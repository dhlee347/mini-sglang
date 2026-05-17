# minisgl.message — 프로세스 간 메시지 시스템

## 역할

ZMQ 소켓을 통해 API 서버, 토크나이저, 스케줄러, 디토크나이저 사이를 오가는 모든 메시지 타입을 정의합니다. msgpack 기반 자동 직렬화/역직렬화를 지원합니다.

**파일**:
- `python/minisgl/message/frontend.py` — API 서버 ↔ 토크나이저/디토크나이저 방향 메시지
- `python/minisgl/message/backend.py` — 토크나이저 → 스케줄러 방향 메시지
- `python/minisgl/message/tokenizer.py` — 토크나이저 워커 내부 메시지 래퍼
- `python/minisgl/message/utils.py` — 직렬화/역직렬화 유틸리티

---

## 메시지 흐름 개요

```
[API Server]
    │  TokenizeMsg / AbortMsg
    ▼
[Tokenizer Worker]
    │  UserMsg / AbortBackendMsg
    ▼
[Scheduler (Rank 0)]
    │  DetokenizeMsg
    ▼
[Detokenizer Worker]
    │  UserReply
    ▼
[API Server]
```

---

## 파일별 설명

### `frontend.py` — API 서버 → 토크나이저 / 디토크나이저 → API 서버

#### `BaseTokenizerMsg` (API 서버 → 토크나이저)
```python
TokenizeMsg(uid, text, sampling_params)   # 텍스트 → 토큰 변환 요청
AbortMsg(uid)                              # 요청 취소
BatchTokenizerMsg(data: List)             # 여러 메시지 묶음
```

- `text` 필드: `str` (raw prompt) 또는 `List[dict]` (chat messages 형식)

#### `BaseFrontendMsg` (디토크나이저 → API 서버)
```python
UserReply(uid, incremental_output, finished)  # 스트리밍 텍스트 조각
BatchFrontendMsg(data: List[UserReply])       # 여러 UserReply 묶음
```

---

### `backend.py` — 토크나이저 → 스케줄러

```python
UserMsg(uid, input_ids, sampling_params)   # 토큰화 완료된 요청
AbortBackendMsg(uid)                        # 스케줄러에게 취소 전달
ExitMsg()                                   # 스케줄러 종료 신호
BatchBackendMsg(data: List)                # 여러 메시지 묶음
```

- `input_ids`: CPU 1D int32 tensor (ZMQ로 전송 전 numpy 변환 후 직렬화)

#### `DetokenizeMsg` (스케줄러 → 디토크나이저)
```python
DetokenizeMsg(uid, next_token, finished)   # 생성된 토큰 하나
```

---

### `utils.py` — 직렬화/역직렬화 엔진

타입 이름을 `__type__` 필드로 저장해 재귀적으로 직렬화합니다.

**지원 타입**:
- 기본 Python 타입 (`int`, `float`, `str`, `bool`, `None`, `bytes`)
- `list`, `tuple`, `dict`
- `torch.Tensor` — 1D 텐서를 numpy bytes + dtype으로 직렬화
- dataclass — `__dict__`를 재귀 직렬화

```python
serialize_type(obj)           # 객체 → dict
deserialize_type(cls_map, d)  # dict → 객체 (cls_map으로 타입 복원)
```

---

## 배치 처리 최적화

단일 메시지와 배치 메시지를 구분하여 직렬화 오버헤드를 줄입니다.

```python
# 보내는 쪽 (tokenizer/server.py)
if len(batch) == 1:
    send(batch[0])           # 단일 메시지로 전송
else:
    send(BatchFrontendMsg(batch))  # 묶음으로 전송

# 받는 쪽
def _unwrap_msg(msg):
    if isinstance(msg, BatchFrontendMsg):
        return msg.data      # 풀어서 처리
    return [msg]
```

---

## ZMQ 소켓 주소 체계

프로세스 ID 기반 suffix를 붙여 동일 머신에서 여러 인스턴스가 충돌하지 않게 합니다.

```
ipc:///tmp/minisgl_0.pid=<PID>   # 토크나이저 → 스케줄러
ipc:///tmp/minisgl_1.pid=<PID>   # 스케줄러 → 디토크나이저
ipc:///tmp/minisgl_2.pid=<PID>   # 스케줄러 → 스케줄러 브로드캐스트 (TP > 1)
ipc:///tmp/minisgl_3.pid=<PID>   # 디토크나이저 → API 서버
ipc:///tmp/minisgl_4.pid=<PID>   # 별도 토크나이저 프로세스용 (--num-tokenizer > 0)
```
