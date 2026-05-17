# Mini-SGLang

SGLang의 경량 구현체로, ~5,000줄의 Python으로 작성된 LLM 추론 프레임워크.  
원본: https://github.com/sgl-project/mini-sglang

## 빠른 실행

```bash
# 설치
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .

# 서버 실행 (기본 포트 1919, OpenAI 호환 API)
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 멀티 GPU (Tensor Parallelism)
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4

# 인터랙티브 셸
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | (필수) | 모델 경로 또는 HuggingFace repo ID |
| `--tp` | 1 | Tensor Parallelism 수 |
| `--port` | 1919 | 서버 포트 |
| `--attn` | auto | attention 백엔드 (`fa`, `fi`, `trtllm`, `fa,fi`) |
| `--cache` | radix | KV 캐시 전략 (`radix`, `naive`) |
| `--page-size` | 16 | KV 캐시 페이지 크기 |
| `--max-prefill-length` | 8192 | Chunked Prefill 청크 크기 |
| `--cuda-graph-max-bs` | auto | CUDA Graph 최대 배치 크기 (0=비활성화) |
| `--memory-ratio` | 0.9 | GPU 메모리 중 KV 캐시 비율 |
| `--model-source` | huggingface | 모델 다운로드 소스 (`huggingface`, `modelscope`) |

환경변수: `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` — overlap scheduling 비활성화

## 아키텍처

멀티 프로세스 분산 시스템으로, ZeroMQ(제어 메시지)와 NCCL/PyNCCL(GPU 텐서 통신)로 프로세스 간 통신.

```
User → API Server → Tokenizer → Scheduler(Rank 0) ↔ Scheduler(Rank 1..N)
                                       ↓                      ↓
                                   Engine(GPU 0)         Engine(GPU N)
                  Detokenizer ← Scheduler(Rank 0)
                       ↓
                   API Server → User
```

**Overlap Scheduling**: 현재 배치 GPU 실행과 이전 배치 결과 처리를 겹쳐 CPU 오버헤드 은닉.

## 코드 구조 (`python/minisgl/`)

```
minisgl/
├── core.py              # Req, Batch, Context, SamplingParams 핵심 데이터클래스
├── env.py               # 환경변수 설정 (MINISGL_*)
├── shell.py             # 인터랙티브 셸 구현
│
├── server/              # 서버 진입점
│   ├── args.py          # CLI 인자 파싱 (ServerArgs)
│   ├── launch.py        # 서브프로세스 실행 (launch_server)
│   └── api_server.py    # FastAPI 서버 (/v1/chat/completions)
│
├── scheduler/           # 스케줄러 (TP 워커당 1개)
│   ├── scheduler.py     # Scheduler 메인 루프, overlap/normal 스케줄링
│   ├── prefill.py       # PrefillManager, ChunkedReq
│   ├── decode.py        # DecodeManager
│   ├── cache.py         # CacheManager (radix/naive 선택)
│   ├── table.py         # TableManager (page table 슬롯 관리)
│   ├── io.py            # SchedulerIOMixin (ZMQ 통신)
│   └── config.py        # SchedulerConfig
│
├── engine/              # 단일 GPU 추론 엔진
│   ├── engine.py        # Engine (모델+KVCache+그래프 통합 관리)
│   ├── config.py        # EngineConfig
│   ├── graph.py         # GraphRunner (CUDA Graph 캡처/재실행)
│   └── sample.py        # Sampler (greedy/top-k/top-p)
│
├── models/              # LLM 모델 구현
│   ├── llama.py         # LLaMA-3
│   ├── qwen3.py         # Qwen3 dense
│   ├── qwen3_moe.py     # Qwen3 MoE
│   ├── qwen2.py         # Qwen2.5
│   ├── mistral.py       # Mistral
│   ├── weight.py        # HuggingFace 가중치 로딩/샤딩
│   └── register.py      # 모델 레지스트리
│
├── attention/           # Attention 백엔드
│   ├── base.py          # BaseAttnBackend, BaseAttnMetadata
│   ├── fa.py            # FlashAttention 백엔드
│   ├── fi.py            # FlashInfer 백엔드
│   └── trtllm.py        # TensorRT-LLM FMHA 백엔드
│
├── kvcache/             # KV 캐시 관리
│   ├── base.py          # BaseKVCachePool, BaseCacheHandle
│   ├── mha_pool.py      # MHAKVCache (실제 GPU 텐서 풀)
│   ├── radix_cache.py   # RadixCacheManager (prefix 재사용)
│   └── naive_cache.py   # NaiveCacheManager
│
├── layers/              # TP 지원 레이어
│   ├── linear.py        # ColumnParallelLinear, RowParallelLinear
│   ├── attention.py     # AttentionLayer
│   ├── embedding.py     # VocabParallelEmbedding
│   ├── norm.py          # RMSNorm
│   ├── rotary.py        # RoPE
│   ├── moe.py           # MoE 레이어
│   └── base.py          # 공통 베이스 클래스
│
├── moe/                 # MoE 백엔드
│   ├── base.py          # BaseMoeBackend
│   └── fused.py         # FusedMoE (triton 기반)
│
├── distributed/         # TP 통신 인터페이스
│   ├── impl.py          # all-reduce, all-gather 구현
│   └── info.py          # DistributedInfo
│
├── message/             # ZMQ 메시지 타입 (msgpack 직렬화)
│   ├── frontend.py      # API Server ↔ Tokenizer 메시지
│   ├── backend.py       # Tokenizer ↔ Scheduler 메시지
│   └── tokenizer.py     # Tokenizer 내부 메시지
│
├── tokenizer/           # 토크나이저 워커
│   ├── tokenize.py      # tokenize_worker
│   └── detokenize.py    # detokenize_worker
│
├── kernel/              # 커스텀 CUDA 커널
│   ├── csrc/            # C++/CUDA 소스 (tvm-ffi 바인딩)
│   │   ├── jit/         # JIT 컴파일 커널 (index.cu, store.cu)
│   │   └── src/         # 런타임 커널 (pynccl.cu, radix.cpp, tensor.cpp)
│   ├── pynccl.py        # Python NCCL 래퍼
│   ├── radix.py         # RadixTree C++ 바인딩
│   ├── store.py         # KV store 커널
│   ├── index.py         # 인덱싱 커널
│   └── triton/          # Triton 커널
│       └── fused_moe.py # Fused MoE 커널
│
├── llm/
│   └── llm.py           # LLM 클래스 (Python API 인터페이스)
│
├── benchmark/           # 벤치마크 유틸리티
│   ├── client.py        # API 클라이언트
│   └── perf.py          # 성능 측정
│
└── utils/               # 공용 유틸리티
    ├── logger.py        # 로거 (rank-aware)
    ├── mp.py            # 멀티프로세싱 유틸
    ├── registry.py      # Registry 패턴 헬퍼
    └── torch_utils.py   # torch 유틸
```

## 핵심 데이터 흐름

- `Req`: 단일 요청 상태 (`input_ids`, `cached_len`, `device_len`, `table_idx`, `cache_handle`)
- `Batch`: prefill/decode 배치 (`reqs`, `input_ids`, `positions`, `out_loc`, `attn_metadata`)
- `Context`: 전역 추론 컨텍스트 (`page_table`, `kv_cache`, `attn_backend`, `moe_backend`)
- `ForwardInput`: overlap scheduling용 배치 입력 캐시 (`batch`, `sample_args`, `input_tuple`, `write_tuple`)

## 개발 환경

```bash
# dev 의존성 설치
uv pip install -e ".[dev]"

# pre-commit 훅 설치
pre-commit install

# 테스트
pytest tests/

# 린터 (ruff + black, line-length=100)
ruff check python/
black python/
```

**테스트 위치**: `tests/core/`, `tests/kernel/`, `tests/misc/`  
**벤치마크**: `benchmark/offline/bench.py`, `benchmark/online/bench_qwen.py`

## 지원 모델

- LLaMA-3 시리즈
- Qwen3 시리즈 (dense + MoE)
- Qwen2.5 시리즈
- Mistral

새 모델 추가: `python/minisgl/models/`에 구현 후 `register.py`에 등록.

## 주요 특징 구현 위치

| 기능 | 파일 |
|------|------|
| Radix Cache | `kvcache/radix_cache.py`, `kernel/radix.py` |
| Chunked Prefill | `scheduler/prefill.py` |
| Overlap Scheduling | `scheduler/scheduler.py` (`overlap_loop`) |
| CUDA Graph | `engine/graph.py` |
| Tensor Parallelism | `distributed/`, `layers/` |
| PyNCCL | `kernel/pynccl.py`, `kernel/csrc/src/pynccl.cu` |
