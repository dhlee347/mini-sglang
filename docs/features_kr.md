# Mini-SGLang 기능

## 온라인 서빙

Mini-SGLang은 OpenAI 호환 API 서버를 통한 온라인 서빙을 지원합니다. 표준 `/v1/chat/completions` 엔드포인트를 제공하여 기존 도구 및 클라이언트와 원활하게 연동할 수 있습니다. 자세한 커맨드라인 인자 및 설정 옵션은 `python -m minisgl --help`를 실행하여 확인하세요.

## 인터랙티브 셸 모드

데모 및 테스트 목적으로 인터랙티브 셸 모드를 사용할 수 있습니다. 이 모드에서는 프롬프트를 직접 입력하면 LLM이 실시간으로 응답을 생성합니다. 셸은 문맥 유지를 위해 대화 기록을 자동으로 캐싱합니다. 대화 기록을 초기화하고 새 세션을 시작하려면 `/reset` 명령어를 사용하세요.

예시:

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 분산 서빙

여러 GPU에 걸쳐 성능을 확장하기 위해 Mini-SGLang은 텐서 병렬처리(Tensor Parallelism, TP)를 지원합니다. `--tp n` 인자로 GPU 수를 지정하여 분산 서빙을 활성화할 수 있으며, 여기서 `n`은 병렬처리 정도입니다.

## 지원 모델

현재 다음 dense 모델 아키텍처를 지원합니다:

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) 시리즈
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) 시리즈 (MoE 포함)
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) 시리즈

## 청크 프리필 (Chunked Prefill)

[Sarathi-Serve](https://arxiv.org/abs/2403.02310)에서 제안된 청크 프리필은 기본적으로 활성화되어 있습니다. 이 기능은 프리필 단계에서 긴 프롬프트를 더 작은 청크로 분할하여 피크 메모리 사용량을 크게 줄이고 장문 컨텍스트 서빙 시 OOM(Out-Of-Memory) 오류를 방지합니다. 청크 크기는 `--max-prefill-length n`으로 설정할 수 있습니다. 단, `n`을 매우 작은 값(예: 128)으로 설정하면 성능이 크게 저하될 수 있으므로 권장하지 않습니다.

## 페이지 크기

`--page-size` 인자를 사용하여 시스템의 페이지 크기를 지정할 수 있습니다.

## 어텐션 백엔드

Mini-SGLang은 [`FlashAttention`](https://github.com/Dao-AILab/flash-attention)(`fa`), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer)(`fi`), [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM)(`trtllm`) 등 고성능 어텐션 커널을 통합합니다. 효율을 극대화하기 위해 프리필과 디코드 단계에 서로 다른 백엔드를 사용할 수 있습니다. 예를 들어 NVIDIA Hopper GPU에서는 기본적으로 프리필에 `FlashAttention 3`, 디코드에 `FlashInfer`를 사용합니다.

`--attn` 인자로 백엔드를 지정할 수 있습니다. 두 값을 제공하면(예: `--attn fa,fi`) 첫 번째는 프리필 백엔드, 두 번째는 디코드 백엔드로 사용됩니다. 일부 어텐션 백엔드는 사용자 지정 페이지 크기를 재정의할 수 있습니다(예: `trtllm`은 페이지 크기 16, 32, 64만 지원).

## CUDA 그래프

디코딩 중 CPU 실행 오버헤드를 최소화하기 위해 Mini-SGLang은 CUDA 그래프 캡처 및 재실행을 지원합니다. 이 기능은 기본적으로 활성화되어 있습니다. CUDA 그래프 캡처의 최대 배치 크기는 `--cuda-graph-max-bs n`으로 설정할 수 있으며, `n`을 `0`으로 설정하면 이 기능이 비활성화됩니다.

## 래딕스 캐시 (Radix Cache)

[SGLang](https://github.com/sgl-project/sglang.git)의 원래 설계를 채택한 Mini-SGLang은 KV(Key-Value) 캐시를 관리하기 위해 래딕스 캐시를 구현합니다. 이를 통해 요청 간 공유 프리픽스에 대한 KV 캐시를 재사용하여 중복 연산을 줄입니다. 이 기능은 기본적으로 활성화되어 있으며, `--cache naive`를 사용하여 단순 캐시 관리 전략으로 전환할 수 있습니다.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*[LMSYS 블로그](https://lmsys.org/blog/2024-01-17-sglang/)의 Radix Attention 일러스트레이션.*

## 오버랩 스케줄링 (Overlap Scheduling)

CPU 오버헤드를 더욱 줄이기 위해 Mini-SGLang은 [NanoFlow](https://arxiv.org/abs/2408.12757)에서 제안된 오버랩 스케줄링 기법을 적용합니다. 이 방식은 CPU 스케줄링 오버헤드를 GPU 연산과 겹쳐 전체 시스템 처리량을 향상시킵니다.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*[LMSYS 블로그](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)의 오버랩 스케줄링 일러스트레이션.*
