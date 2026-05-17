# minisgl.llm — 오프라인 Python API

## 역할

서버 없이 Python 코드에서 직접 Mini-SGLang을 사용할 수 있는 고수준 인터페이스입니다. `Scheduler`를 상속해 ZMQ 대신 메모리 내 통신으로 오프라인 배치 추론을 수행합니다.

**파일**: `python/minisgl/llm/llm.py`

---

## `LLM` 클래스

```python
class LLM(Scheduler):
    def __init__(self, model_path, dtype=torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),   # 단일 GPU만 지원
            dtype=dtype,
            offline_mode=True,               # ZMQ 비활성화
            **kwargs,
        )
        super().__init__(config)
```

`offline_mode=True`로 초기화하면 `SchedulerIOMixin`의 `receive_msg` / `send_result`가 오버라이드 메서드로 교체됩니다.

---

## 주요 메서드

### `generate(prompts, sampling_params)`

배치 추론 실행:

```python
llm = LLM("Qwen/Qwen3-0.6B")
results = llm.generate(
    prompts=["Hello, my name is", "The capital of Korea is"],
    sampling_params=SamplingParams(max_tokens=100, temperature=0.8)
)
# results = [{"text": "...", "token_ids": [...]}, ...]
```

내부 흐름:
```python
1. pending_requests에 (prompt, sampling_params) 저장
2. run_forever() 실행
3. RequestAllFinished 예외로 루프 종료
4. status_map에서 결과 수집 → tokenizer.decode(output_ids)
```

### `offline_receive_msg(blocking)`

ZMQ 대신 `pending_requests`에서 UserMsg를 생성합니다.

```python
# prefill_budget 내에서 요청들을 UserMsg로 변환
for prompt, sp in pending_requests[:budget]:
    input_ids = tokenize(prompt)
    uid = counter++
    yield UserMsg(uid, input_ids, sp)
```

`blocking=True`이고 `pending_requests`가 비어있으면 `RequestAllFinished` 예외 발생 → `run_forever()` 루프 종료.

### `offline_send_result(reply)`

스케줄러 결과를 ZMQ 대신 `status_map`에 기록합니다.

```python
for msg in reply:
    status = status_map[msg.uid]
    if not (msg.finished and msg.next_token == eos_token_id):
        status.output_ids.append(msg.next_token)
```

---

## `RequestStatus`

```python
@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]   # 생성된 토큰 누적
```

---

## 사용 예시

```python
from minisgl.llm import LLM
from minisgl.core import SamplingParams

llm = LLM(
    model_path="Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    page_size=16,
    cache_type="radix",
)

results = llm.generate(
    prompts=["Tell me about Seoul.", "What is Python?"],
    sampling_params=SamplingParams(max_tokens=256, temperature=0.7, top_p=0.9),
)

for r in results:
    print(r["text"])
```

---

## Scheduler와의 차이점

| 항목 | `Scheduler` (온라인) | `LLM` (오프라인) |
|------|---------------------|----------------|
| 통신 | ZMQ 소켓 | 메모리 내 직접 |
| 토크나이징 | 별도 프로세스 | 인라인 처리 |
| 디토크나이징 | 별도 프로세스 | `tokenizer.decode()` 후처리 |
| TP | 멀티 프로세스 지원 | 단일 GPU만 |
| 스트리밍 | 지원 (SSE) | 미지원 (배치 완료 후 반환) |
