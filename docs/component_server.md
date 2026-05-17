# minisgl.server — API 서버 & 프로세스 런처

## 역할

사용자가 HTTP 요청을 보내는 진입점입니다. OpenAI 호환 REST API를 제공하고, 백엔드 서브프로세스(스케줄러, 토크나이저) 전체를 초기화·관리합니다.

**파일**:
- `python/minisgl/server/args.py` — CLI 인자 파싱 및 `ServerArgs` dataclass
- `python/minisgl/server/launch.py` — 전체 서브프로세스 실행 (`launch_server`)
- `python/minisgl/server/api_server.py` — FastAPI 앱 및 비동기 요청 처리

---

## `args.py` — 설정 관리

### `ServerArgs`
`SchedulerConfig`를 상속하는 frozen dataclass. 서버 전용 추가 필드:

```python
server_host: str = "127.0.0.1"
server_port: int = 1919
num_tokenizer: int = 0    # 0이면 tokenizer/detokenizer를 같은 프로세스에서 실행
silent_output: bool = False
```

**ZMQ 주소 프로퍼티들**: `zmq_frontend_addr`, `zmq_tokenizer_addr`, `zmq_backend_addr` 등 소켓 주소를 자동 계산.

### `parse_args(args)`
`sys.argv`를 파싱해 `(ServerArgs, run_shell)` 튜플 반환.
- `--model` / `--model-path`: HuggingFace repo ID 또는 로컬 경로
- `--tp`: Tensor Parallelism 크기
- `--dtype auto`: HF config에서 dtype 자동 추론
- `--model-source modelscope`: ModelScope에서 가중치 다운로드
- `--shell` 플래그: `cuda_graph_max_bs=1`, `max_running_req=1`로 강제

---

## `launch.py` — 서브프로세스 런처

### `launch_server(run_shell)`

1. `parse_args()`로 설정 파싱
2. `run_api_server(config, start_subprocess, run_shell)` 호출

### `start_subprocess()` 콜백
API 서버 uvicorn 시작 전 백그라운드 서브프로세스를 `spawn` 방식으로 실행:

```
mp.Process × world_size   → _run_scheduler(args[rank])
mp.Process × 1            → tokenize_worker(detokenizer)
mp.Process × num_tokenizer → tokenize_worker(tokenizer_i)
```

`ack_queue`를 통해 `num_tokenizer + 2`개의 ready 신호를 기다린 후 HTTP 서버를 연다.

### `_run_scheduler(args, ack_queue)`
각 TP 랭크에서 실행. Rank 0만 `ack_queue`에 ready 신호 전송. `scheduler.run_forever()` 루프 실행.

---

## `api_server.py` — FastAPI 비동기 요청 처리

### `FrontendManager`

API 서버 핵심 상태 관리 클래스. 요청별 asyncio 이벤트 기반 스트리밍.

```python
uid_counter: int                       # 단조 증가 요청 ID
ack_map: Dict[int, List[UserReply]]    # uid → 수신된 응답 조각들
event_map: Dict[int, asyncio.Event]    # uid → 응답 도착 이벤트
send_tokenizer: ZmqAsyncPushQueue      # 토크나이저로 전송
recv_tokenizer: ZmqAsyncPullQueue      # 디토크나이저에서 수신
```

**핵심 메서드**:

| 메서드 | 설명 |
|--------|------|
| `new_user()` | 새 uid 할당, ack_map/event_map 초기화 |
| `listen()` | 백그라운드 태스크 — 디토크나이저 수신 루프 |
| `wait_for_ack(uid)` | asyncio Event로 UserReply 스트리밍 |
| `stream_generate(uid)` | `/generate` SSE 스트림 생성기 |
| `stream_chat_completions(uid)` | `/v1/chat/completions` SSE 스트림 생성기 |
| `stream_with_cancellation(gen, req, uid)` | 클라이언트 연결 해제 감지 후 abort |
| `abort_user(uid)` | `AbortMsg(uid)` 전송으로 스케줄러에 취소 요청 |

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/generate` | raw 텍스트 입력, SSE 스트리밍 출력 |
| POST | `/v1/chat/completions` | OpenAI chat completion 형식 |
| GET | `/v1/models` | 현재 모델 정보 |
| GET/POST | `/v1` | 헬스체크 |

### 요청 처리 흐름 (스트리밍)

```
HTTP 요청 수신
    ↓
FrontendManager.new_user() → uid 발급
    ↓
TokenizeMsg(uid, text, sampling_params) → ZMQ push
    ↓
StreamingResponse 반환 (SSE)
    ↓ (비동기로 계속)
listen() → UserReply 수신 → ack_map[uid] 저장 → event_map[uid].set()
    ↓
wait_for_ack(uid) → UserReply 꺼내서 SSE 청크로 변환
    ↓
reply.finished == True → 스트림 종료 + "data: [DONE]"
```

---

## 셸 모드 (`shell()`)

`--shell` 플래그 시 uvicorn 대신 `asyncio.run(shell())` 실행.
- `prompt_toolkit` 기반 인터랙티브 입력 (`/exit`, `/reset` 명령)
- 대화 히스토리를 messages 리스트로 누적
- 내부적으로 `shell_completion()` → `stream_generate()` 호출
- 종료 시 `psutil`로 자식 프로세스 전체 kill
