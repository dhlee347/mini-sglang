# RadixAttention과 SGLang을 이용한 빠르고 표현력 있는 LLM 추론

> 원문: [LMSYS Blog — Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/) (2024-01-17)  
> 저자: Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng

---

## 개요

LLM(대규모 언어 모델)은 여러 차례의 생성 호출, 고급 프롬프팅 기법, 제어 흐름, 외부 환경과의 상호작용이 필요한 복잡한 작업에 점점 더 많이 활용되고 있습니다. 그러나 기존 시스템은 이러한 복잡한 LLM 프로그램을 효율적으로 처리하지 못합니다.

이 글에서는 이 문제를 해결하기 위해 설계된 시스템인 **SGLang**을 소개합니다. SGLang은 두 가지 핵심 구성 요소로 이루어져 있습니다.

- **프론트엔드**: LLM 프로그래밍을 위한 파이썬 내장 도메인 특화 언어
- **백엔드**: KV 캐시 재사용을 위한 RadixAttention 등 최적화 기술

벤치마크 결과, SGLang은 기존 시스템(Guidance, vLLM)보다 **최대 5배 높은 처리량**을 달성했습니다.

---

## 프론트엔드: SGLang으로 쉬운 LLM 프로그래밍

SGLang은 고급 프롬프팅 기법, 제어 흐름, 멀티모달, 디코딩 제약, 외부 상호작용을 쉽게 표현할 수 있게 해줍니다. OpenAI, Anthropic, Gemini, 로컬 모델 등 다양한 백엔드에서 실행할 수 있습니다.

### 핵심 API 기능

다음은 에세이 심사 예제에서 사용된 주요 기능입니다(branch-solve-merge 기법):

1. **`fork`**: 프롬프트의 병렬 복사본을 여러 개 생성합니다.
2. **`gen`**: LLM 생성을 비동기적으로 호출하고 결과를 변수에 저장합니다.
3. **`[변수명]`**: 생성 결과를 가져옵니다.
4. **`choices`**: 생성에 제약을 부여합니다.
5. **`run`**: SGLang 함수를 인자와 함께 실행합니다.

### 실행 모드

프로그램은 두 가지 방식으로 실행할 수 있습니다.

- **인터프리터 모드**: 즉각적인 실행, 프로그램 내 병렬 처리 지원
- **컴파일러 모드**: 데이터 흐름 그래프로 추적하여 코드 이동, 자동 튜닝 등의 최적화 수행

---

## 백엔드: RadixAttention을 이용한 자동 KV 캐시 재사용

### 문제 인식

SGLang 런타임을 개발하면서, 저자들은 복잡한 LLM 프로그램에서 기존 시스템이 제대로 처리하지 못하는 중요한 최적화 기회를 발견했습니다. 바로 **KV 캐시 재사용**입니다.

동일한 prefix를 가진 서로 다른 프롬프트들은 중간 KV 캐시를 공유할 수 있어 불필요한 메모리와 연산을 피할 수 있습니다. 이는 다음과 같은 상황에서 특히 유용합니다.

- 멀티턴 대화
- 퓨샷 학습(few-shot learning)
- Tree-of-Thought 추론
- 자기 일관성 샘플링(self-consistency sampling)
- 공유 시스템 프롬프트가 있는 다중 사용자 요청

### RadixAttention 소개

**RadixAttention**은 여러 LLM 생성 호출에 걸쳐 KV 캐시를 자동으로 효율적으로 재사용하는 기술입니다. 기존 시스템과 달리, 생성 완료 후 KV 캐시를 버리지 않고 프롬프트와 생성 결과 모두의 KV 캐시를 **래딕스 트리(Radix Tree)** 에 보존합니다.

> 래딕스 트리(Radix Tree)는 트라이(Trie, 접두사 트리)의 공간 효율적인 대안으로, 엣지에 가변 길이의 원소 시퀀스 레이블을 붙일 수 있는 자료 구조입니다.

### 기술 구현

#### 데이터 구조

래딕스 트리는 다음의 매핑을 관리합니다.

- **키(Key)**: 토큰 시퀀스 (입력 prefix)
- **값(Value)**: 대응하는 KV 캐시 텐서 (GPU의 페이지드 레이아웃에 저장, 토큰당 1페이지)

트리 구조 자체는 **CPU에 저장**되므로 유지 관리 오버헤드가 최소화됩니다.

#### 메모리 관리

- **LRU 축출 정책(LRU Eviction Policy)**: 재귀적으로 리프 노드부터 축출합니다.
  - 각 노드는 현재 해당 노드를 사용 중인 요청 수를 추적하는 **참조 카운터(reference counter)** 를 유지합니다.
  - 참조 카운터가 0이 되면 해당 노드는 축출 대상이 됩니다.
- **기존 기술과의 호환성**: 연속 배칭(continuous batching), 페이지드 어텐션(paged attention), 텐서 병렬처리(tensor parallelism)와 완벽하게 호환됩니다.
- **오버헤드 없음**: 캐시 히트가 없는 경우에도 무시할 수 있는 수준의 오버헤드만 발생합니다(ShareGPT 데이터셋에서 0.3% 미만 측정).

#### 캐시 인식 스케줄링

선입선출(FIFO) 방식 대신 **매칭된 prefix 길이 기준으로 요청 우선순위를 부여**합니다. 이 '가장 긴 공유 prefix 우선(Longest-Shared-Prefix-First)' 방식은 래딕스 트리의 깊이 우선 탐색(DFS)과 유사하며, 캐시 크기가 최대 요청 길이를 초과하는 경우 캐시 히트율을 최적화하는 것이 증명되어 있습니다.

---

### RadixAttention 동작 예시 (9단계)

아래 그림은 시간 순서에 따른 래딕스 트리의 변화를 보여줍니다.

![RadixAttention 9단계 동작 과정](https://lmsys.org/images/blog/sglang/radix_attn.jpg)

| 단계 | 상황 | 설명 |
|------|------|------|
| **1** | 초기 상태 | 빈 래딕스 트리 |
| **2** | 첫 번째 대화 | 시스템 프롬프트, 사용자 메시지 "Hello", 응답 "Hi"를 처리하여 하나의 엣지로 병합 |
| **3** | Prefix 재사용 | 새 프롬프트가 기존 prefix와 일치함을 발견, 해당 KV 캐시를 재사용하고 새 내용을 덧붙임 |
| **4** | 세션 분기 | 새 채팅 세션 시작, 노드 분할(node splitting)을 통해 두 세션이 시스템 프롬프트를 공유 |
| **5** | 메모리 압박하 축출 | 메모리 한계로 인해 특정 노드 축출, 새 내용은 다른 위치에 추가 |
| **6** | 퓨샷 쿼리 삽입 | 새 퓨샷 학습 쿼리 수신, prefix가 일치하지 않아 루트 노드 분할 |
| **7** | 배치 처리 | 동일한 예시를 공유하는 추가 퓨샷 쿼리 묶음 처리, 노드 분할로 공유 |
| **8** | LRU 정리 | 첫 번째 세션의 새 메시지 처리를 위해 최근에 사용되지 않은 두 번째 세션 노드 축출 |
| **9** | 자기 일관성 샘플링 | 추가 답변 샘플링 요청 처리를 위해 중간 노드 축출 |

---

## 평가 결과

### 실험 설정

| 항목 | 설정 |
|------|------|
| **소형 모델** | Llama-7B, NVIDIA A10G GPU 1장 (24GB) |
| **대형 모델** | Mixtral-8x7B, NVIDIA A10G GPU 8장 (텐서 병렬처리) |
| **정밀도** | FP16 |
| **비교 대상** | vLLM v0.2.5, Guidance v0.1.8, HuggingFace TGI v1.3.0 |

### 벤치마크 워크로드 (9가지)

- MMLU (5-shot 퓨샷)
- HellaSwag (20-shot 퓨샷)
- ReAct Agent
- Tree-of-Thought 추론
- JSON 디코딩
- 멀티턴 채팅 (짧은/긴 출력)
- DSPy RAG 파이프라인
- LLaVA 비전 벤치마크

### 주요 결과

| 지표 | 수치 |
|------|------|
| 최대 처리량 향상 | **최대 6.4배** |
| 최대 지연시간 감소 | **최대 3.7배** |
| 캐시 히트율 범위 | 50~99% (벤치마크별 상이) |
| 실제 서비스 (LLaVA-Next-34B) | 히트율 52.4% |
| 실제 서비스 (Vicuna-33B) | 히트율 74.1%, 첫 토큰 지연시간 1.7배 감소 |

- **MMLU, HellaSwag**: 2단계 공유 구조(system prompt + few-shot examples)에서 큰 이점
- **Tree-of-Thought, Skeleton-of-Thought**: 프로그램 내 병렬 처리로 추가 가속
- **멀티턴 채팅**: 짧은 출력에서 1.7배 처리량 향상
- **LLaVA 멀티모달**: 이미지 토큰 KV 캐시 재사용으로 최대 6배 처리량 향상

### 실제 서비스 적용

- **LLaVA 온라인 데모** 서비스
- **DSPy 프레임워크** 통합 백엔드

---

## 자원

- **코드**: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
- **논문**: [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)

## 인용

```bibtex
@misc{zheng2023efficiently,
      title={Efficiently Programming Large Language Models using SGLang},
      author={Lianmin Zheng and Liangsheng Yin and Zhiqiang Xie and
              Jeff Huang and Chuyue Sun and Cody Hao Yu and Shiyi Cao and
              Christos Kozyrakis and Ion Stoica and Joseph E. Gonzalez and
              Clark Barrett and Ying Sheng},
      year={2023},
      eprint={2312.07104},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
