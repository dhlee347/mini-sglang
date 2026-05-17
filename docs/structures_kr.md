# Mini-SGLang 구조

## 시스템 아키텍처

Mini-SGLang은 LLM(대규모 언어 모델) 추론을 효율적으로 처리하기 위한 분산 시스템으로 설계되었습니다. 여러 독립적인 프로세스가 협력하여 동작합니다.

### 핵심 구성 요소

- **API 서버**: 사용자 진입점. OpenAI 호환 API(예: `/v1/chat/completions`)를 제공하여 프롬프트를 받고 생성된 텍스트를 반환합니다.
- **토크나이저 워커**: 입력 텍스트를 모델이 이해할 수 있는 숫자(토큰)로 변환합니다.
- **디토크나이저 워커**: 모델이 생성한 숫자(토큰)를 사람이 읽을 수 있는 텍스트로 변환합니다.
- **스케줄러 워커**: 핵심 워커 프로세스. 멀티 GPU 설정에서는 각 GPU(이를 **TP 랭크**라 함)마다 하나의 스케줄러 워커가 존재합니다. 해당 GPU의 연산 및 리소스 할당을 관리합니다.

### 데이터 흐름

구성 요소들은 제어 메시지에 **ZeroMQ(ZMQ)** 를, GPU 간 대용량 텐서 데이터 교환에 **NCCL**(`torch.distributed` 경유)을 사용하여 통신합니다.

![프로세스 개요 다이어그램](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**요청 생애주기:**

1. **사용자**가 **API 서버**에 요청을 전송합니다.
2. **API 서버**가 요청을 **토크나이저**로 전달합니다.
3. **토크나이저**가 텍스트를 토큰으로 변환하여 **스케줄러(Rank 0)** 에 전송합니다.
4. **스케줄러(Rank 0)** 가 요청을 다른 모든 스케줄러에 브로드캐스트합니다(멀티 GPU 사용 시).
5. **모든 스케줄러**가 요청을 스케줄링하고 로컬 **엔진**을 트리거하여 다음 토큰을 연산합니다.
6. **스케줄러(Rank 0)** 가 출력 토큰을 수집하여 **디토크나이저**로 전송합니다.
7. **디토크나이저**가 토큰을 텍스트로 변환하여 **API 서버**로 전송합니다.
8. **API 서버**가 결과를 **사용자**에게 스트리밍합니다.

## 코드 구성 (`minisgl` 패키지)

소스 코드는 `python/minisgl`에 위치합니다. 아래는 개발자를 위한 모듈 설명입니다:

- `minisgl.core`: 요청 상태를 나타내는 핵심 데이터클래스 `Req`와 `Batch`, 추론 컨텍스트의 전역 상태를 보유하는 `Context` 클래스, 사용자가 제공하는 샘플링 파라미터를 담는 `SamplingParams` 클래스를 제공합니다.
- `minisgl.distributed`: 텐서 병렬처리에서의 all-reduce 및 all-gather 인터페이스, TP 워커의 TP 정보를 담는 데이터클래스 `DistributedInfo`를 제공합니다.
- `minisgl.layers`: TP 지원을 포함한 LLM 구축을 위한 기본 빌딩 블록(linear, layernorm, embedding, RoPE 등)을 구현합니다. 공통 베이스 클래스는 `minisgl.layers.base`에 정의되어 있습니다.
- `minisgl.models`: Llama, Qwen3 등 LLM 모델을 구현합니다. HuggingFace에서 가중치를 로딩하고 샤딩하는 유틸리티도 포함합니다.
- `minisgl.attention`: 어텐션 백엔드 인터페이스를 제공하고, `flashattention`과 `flashinfer` 백엔드를 구현합니다. `AttentionLayer`에 의해 호출되며 `Context`에 저장된 메타데이터를 사용합니다.
- `minisgl.kvcache`: KVCache 풀과 KVCache 매니저 인터페이스를 제공하고, `MHAKVCache`, `NaiveCacheManager`, `RadixCacheManager`를 구현합니다.
- `minisgl.utils`: 로거 설정 및 ZMQ 래퍼 등 다양한 유틸리티를 제공합니다.
- `minisgl.engine`: 단일 프로세스에서 동작하는 TP 워커인 `Engine` 클래스를 구현합니다. 모델, 컨텍스트, KVCache, 어텐션 백엔드, CUDA 그래프 재실행을 관리합니다.
- `minisgl.message`: API 서버, 토크나이저, 디토크나이저, 스케줄러 간 ZMQ 교환 메시지를 정의합니다. 모든 메시지 타입은 자동 직렬화/역직렬화를 지원합니다.
- `minisgl.scheduler`: 각 TP 워커 프로세스에서 실행되며 해당 `Engine`을 관리하는 `Scheduler` 클래스를 구현합니다. Rank 0 스케줄러는 토크나이저로부터 메시지를 수신하고, 다른 TP 워커의 스케줄러와 통신하며, 디토크나이저로 메시지를 전송합니다.
- `minisgl.server`: CLI 인자와 Mini-SGLang의 모든 서브프로세스를 시작하는 `launch_server`를 정의합니다. `/v1/chat/completions` 등의 엔드포인트를 제공하는 FastAPI 서버를 `minisgl.server.api_server`에 구현합니다.
- `minisgl.tokenizer`: 토크나이즈 및 디토크나이즈 요청을 처리하는 `tokenize_worker` 함수를 구현합니다.
- `minisgl.llm`: Mini-SGLang 시스템과 쉽게 상호작용할 수 있는 Python 인터페이스로 `LLM` 클래스를 제공합니다.
- `minisgl.kernel`: `tvm-ffi`로 Python 바인딩과 JIT 인터페이스를 지원하는 커스텀 CUDA 커널을 구현합니다.
- `minisgl.benchmark`: 벤치마크 유틸리티.
