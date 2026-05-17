# minisgl.tokenizer — 토크나이저 & 디토크나이저 워커

## 역할

텍스트 ↔ 토큰 변환을 담당하는 독립 프로세스입니다. 하나의 워커 프로세스가 토크나이저와 디토크나이저 역할을 모두 수행하거나(`--num-tokenizer 0`), 별도 프로세스로 분리할 수 있습니다.

**파일**:
- `python/minisgl/tokenizer/server.py` — 워커 메인 루프 (`tokenize_worker`)
- `python/minisgl/tokenizer/tokenize.py` — `TokenizeManager` (텍스트 → 토큰)
- `python/minisgl/tokenizer/detokenize.py` — `DetokenizeManager` (토큰 → 텍스트, 스트리밍)

---

## 프로세스 구성

```
num_tokenizer = 0 (기본값):
  프로세스 1개 — tokenizer + detokenizer 통합
  ZMQ PULL: API 서버 및 스케줄러로부터 수신
  ZMQ PUSH: API 서버로 전송 (UserReply)
         + 스케줄러로 전송 (UserMsg)

num_tokenizer = N:
  프로세스 N개 — tokenizer 전용 (TokenizeMsg 처리)
  프로세스 1개 — detokenizer 전용 (DetokenizeMsg 처리)
```

---

## `server.py` — `tokenize_worker()` 메인 루프

```python
def tokenize_worker(
    tokenizer_path, addr, create, backend_addr, frontend_addr,
    local_bs, tokenizer_id, ack_queue
)
```

배치 처리 루프:

```python
while True:
    msg = recv.get()                  # 블로킹 수신
    while len(batch) < local_bs and not recv.empty():
        batch.extend(recv.get())      # 가용 메시지 모두 수집 (배치 최적화)

    detokenize_msgs = [DetokenizeMsg, ...]
    tokenize_msgs   = [TokenizeMsg, ...]
    abort_msgs      = [AbortMsg, ...]

    # DetokenizeMsg → UserReply → API 서버
    replies = detokenize_manager.detokenize(detokenize_msgs)
    send_frontend.put(BatchFrontendMsg(replies))

    # TokenizeMsg → UserMsg → 스케줄러
    tensors = tokenize_manager.tokenize(tokenize_msgs)
    send_backend.put(BatchBackendMsg(tensors))

    # AbortMsg → AbortBackendMsg → 스케줄러
    send_backend.put(BatchBackendMsg([AbortBackendMsg(uid) for ...]))
```

**ZMQ 소켓 방향**:
- PULL `addr`: API 서버(TokenizeMsg) + 스케줄러(DetokenizeMsg) 수신
- PUSH `backend_addr`: 스케줄러로 UserMsg 전송
- PUSH `frontend_addr`: API 서버로 UserReply 전송

---

## `tokenize.py` — `TokenizeManager`

HuggingFace `PreTrainedTokenizerBase`를 래핑합니다.

```python
def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
```

- `msg.text`가 `list` (chat messages): `apply_chat_template(tokenize=False)` → 문자열로 변환 후 인코딩
- `msg.text`가 `str` (raw prompt): 직접 인코딩
- 결과: CPU 1D int32 tensor (`return_tensors="pt"` 후 view(-1))

현재 배치 토크나이징 미지원(TODO 주석). 메시지 단위 순차 처리.

---

## `detokenize.py` — `DetokenizeManager`

스트리밍 디토크나이징을 관리합니다. 불완전한 UTF-8 서로게이트 문자 처리와 단어 경계 처리가 핵심입니다.

### `DecodeStatus`
요청 하나의 디토크나이징 누적 상태:

```python
decoded_ids: List[int]   # 지금까지 받은 모든 토큰 ID
decoded_str: str         # 확정된 텍스트 (서로게이트 없는 부분)
read_offset: int         # batch_decode에 넘긴 토큰 수
surr_offset: int         # 서로게이트 가능성 때문에 보류한 시작점
sent_offset: int         # 이미 API 서버로 전송한 문자 수
```

### `detokenize(msgs)` 처리 흐름

```
1. decoded_ids에 next_token 추가 (EOS & finished는 제외)

2. batch_decode([decoded_ids[surr_offset:]])  → read_str
   batch_decode([decoded_ids[surr_offset:read_offset]]) → surr_str
   # surr_str은 이전 호출에서 보류된 부분

3. new_text = read_str[len(surr_str):]
   if new_text ends with '??' (서로게이트):
       find_printable_text(new_text)  # 공백/한자 경계까지만 전송
   else:
       decoded_str += new_text
       surr_offset = read_offset (확정)

4. incremental_output = decoded_str[sent_offset:]
   sent_offset 갱신
```

### `find_printable_text(text)`
스트리밍 중 불완전한 단어를 잘라내는 휴리스틱:
- `\n`으로 끝남 → 전부 반환
- 마지막/끝에서두번째 문자가 CJK → 전부(또는 마지막 제외) 반환
- 그 외 → 마지막 공백 이후 잘라내기 (`text[:text.rfind(" ") + 1]`)

CJK 판별은 `_is_chinese_char(cp)` — CJK Unified Ideographs 유니코드 블록 범위 검사.

---

## 다른 모듈과의 관계

```
API 서버         → TokenizeMsg  → tokenize_worker
                → AbortMsg     → tokenize_worker
tokenize_worker → UserMsg      → Scheduler (backend)
Scheduler       → DetokenizeMsg → tokenize_worker (detokenizer 소켓)
tokenize_worker → UserReply    → API 서버 (frontend)
```
