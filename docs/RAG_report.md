# RAG 기반 Active Learning 및 Metric Learning 활용 방안 보고서

## 1. 개요

본 보고서는 **CCTV 기반 강의실 출석 확인 Edge AI**를 대상으로, Computer Vision의 **Metric Learning**, **Active Learning**과 Database의 **Vector Retrieval/RAG** 기술을 결합하여 지속적으로 성능을 개선할 수 있는 AI 데이터 파이프라인을 제안한다.

핵심 목표는 CCTV에서 수집되는 다양한 얼굴 데이터 중 **모델이 인식하기 어려운 데이터(Hard Sample)를 자동으로 탐색·관리하고**, 이를 Metric Learning의 학습 데이터로 활용하여 얼굴 인식 모델의 정확도와 강건성을 지속적으로 향상시키는 것이다.

> **핵심 구조:**
> `Vision → Embedding → Vector DB → Retrieval → Active Learning → Labeling → Metric Learning → Model Update`

---

# 2. 핵심 기술

## 2.1 Face Embedding

Face Recognition 모델은 얼굴 이미지를 고정 길이의 벡터로 변환한다.

```text
CCTV Face
    ↓
Face Encoder
    ↓
Embedding Vector
[0.12, -0.35, 0.81, ...]
```

같은 사람의 얼굴은 벡터 공간에서 가깝게, 다른 사람은 멀어지도록 학습한다.

따라서 얼굴 인식 시스템에서 **Embedding은 DB에서 검색할 수 있는 핵심 데이터**가 된다.

---

## 2.2 Metric Learning

Metric Learning은 얼굴 간의 **거리 또는 유사도 자체를 학습**하는 방법이다.

대표적인 방법:

* Contrastive Loss
* Triplet Loss
* ArcFace
* CosFace

특히 CCTV 환경에서는 정면 얼굴뿐 아니라 다음과 같은 다양한 조건을 고려해야 한다.

* 좌·우 측면
* 상·하 방향
* 저조도
* 안경 및 마스크
* 얼굴 가림
* 카메라와의 거리
* 영상 압축 및 Blur

따라서 다양한 조건에서 얻은 얼굴 Embedding이 올바른 거리 관계를 갖도록 학습하는 것이 중요하다.

---

## 2.3 Vector Database 및 Retrieval

얼굴 Embedding을 Vector DB에 저장하면 새로운 얼굴이 입력되었을 때 유사한 얼굴을 검색할 수 있다.

```text
Query Face
    ↓
Embedding
    ↓
Vector DB
    ↓
Similarity Search
    ↓
Top-K Similar Faces
```

검색 결과에는 다음과 같은 메타데이터를 함께 저장한다.

| 데이터        | 내용             |
| ---------- | -------------- |
| Embedding  | 얼굴 특징 벡터       |
| Student ID | 대상 학생          |
| Camera ID  | 촬영 카메라         |
| Timestamp  | 촬영 시간          |
| Pose       | 얼굴 방향          |
| Quality    | 얼굴 품질          |
| Confidence | 인식 신뢰도         |
| Image Path | 원본/Crop 이미지 위치 |

---

## 2.4 RAG

RAG는 **Retrieval-Augmented Generation**의 약자로, 외부 데이터를 검색한 뒤 검색 결과를 생성 모델의 Context로 제공하는 구조이다.

본 시스템에서는 RAG를 **Metric Learning이나 Active Learning 자체를 수행하는 기술로 사용하지 않는다.**

대신,

> **Vector Retrieval + DB Metadata + LLM/VLM**

을 연결하여 학습 데이터 및 모델 성능에 대한 분석을 자동화하는 용도로 활용한다.

예:

> "최근 오인식이 많이 발생한 학생과 촬영 조건을 분석해줘."

↓

```text
Vector DB 검색
+
출석 DB
+
Detection 결과
+
인식 Confidence
+
촬영 환경 Metadata
```

↓

LLM/VLM

↓

> "A 학생은 우측 30도 이상의 측면 및 저조도 환경에서 오인식 빈도가 증가하였다."

---

# 3. 제안 데이터 파이프라인

## 3.1 전체 구조

```text
                CCTV
                  │
                  ▼
          ┌──────────────┐
          │ YOLO / Vision│
          │ Detection    │
          └──────┬───────┘
                 │
                 ▼
             Face Crop
                 │
                 ▼
          Face Recognition
             Encoder
                 │
                 ▼
            Embedding
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
   Vector DB            Metadata DB
       │                   │
       └─────────┬─────────┘
                 ▼
            Retrieval
                 │
                 ▼
        Active Learning
          Sample Score
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
    Easy Data          Hard Sample
                           │
                           ▼
                     Human Labeling
                           │
                           ▼
                    Training Dataset
                           │
                           ▼
                    Metric Learning
                           │
                           ▼
                     Model Update
                           │
                           ▼
                    New Embedding
                           │
                           ▼
                    Vector DB 갱신
```

---

# 4. Active Learning 적용

Active Learning의 목적은 모든 CCTV 데이터를 사람이 라벨링하는 것이 아니라 **학습 가치가 높은 데이터만 선별하는 것**이다.

### 샘플 선정 기준

#### ① Uncertainty

모델의 인식 확률이 낮은 데이터

```text
김철수 0.99 → 학습 우선순위 낮음
김철수 0.53 → 학습 우선순위 높음
```

#### ② Hard Negative

서로 다른 사람인데 Embedding 유사도가 높은 경우

```text
Query = 김철수

김철수   0.94  ← Positive
김영수   0.91  ← Hard Negative
이영희   0.54
```

김철수와 김영수의 샘플은 Metric Learning에 매우 중요한 학습 데이터가 된다.

#### ③ Diversity

동일한 환경의 데이터만 반복해서 선택하지 않고 다음 조건을 다양하게 포함한다.

* 카메라
* 시간대
* 얼굴 방향
* 조명
* 거리
* 가림
* 화질

따라서 **Uncertainty + Hard Negative + Diversity**를 종합하여 Active Learning Score를 계산하는 것을 권장한다.

---

# 5. Metric Learning 적용

Active Learning으로 선별된 데이터를 학습 데이터로 편입한다.

```text
Anchor
   │
   ├── Positive
   │     └─ 같은 학생 / 다른 환경
   │
   └── Negative
         └─ 다른 학생 / 유사한 얼굴
```

특히 Retrieval을 이용하면 **Hard Positive / Hard Negative를 자동으로 구성**할 수 있다.

예:

```text
Anchor: 김철수 정면

Positive:
김철수 측면
김철수 저조도
김철수 원거리

Hard Negative:
김영수 정면
김영수 측면
```

이를 Metric Learning에 활용하면 모델이 단순히 "사람을 구분"하는 것을 넘어 **유사한 사람을 정확하게 분리하는 능력**을 향상시킬 수 있다.

---

# 6. Database 설계

DB는 크게 **정형 데이터와 Vector 데이터**를 분리하는 것을 권장한다.

```text
┌─────────────────────┐
│ Relational DB       │
│                     │
│ 학생 정보            │
│ 수업 정보            │
│ 출석 정보            │
│ 카메라 정보           │
│ 영상 Metadata        │
└─────────┬───────────┘
          │
          │ ID / Metadata
          │
┌─────────▼───────────┐
│ Vector DB           │
│                     │
│ Face Embedding      │
│ Similarity          │
│ Top-K Neighbor      │
│ Hard Sample         │
└─────────────────────┘
```

### 권장 데이터 관계

```text
Student
   │
   ├── Face Images
   │       │
   │       └── Embeddings
   │
   ├── Attendance
   │
   └── Camera / Capture Metadata
```

**Vector DB에는 모든 업무 데이터를 저장하기보다 Embedding 검색이 필요한 데이터 중심으로 저장**하고, 학생·수업·출석 등 정형 정보는 기존 RDB에서 관리하는 것이 효율적이다.

---

# 7. Edge AI 환경 적용

Edge AI에서는 실시간 추론과 학습/검색을 분리하는 것이 중요하다.

## Edge Device

실시간성이 필요한 작업을 담당한다.

```text
CCTV
 ↓
YOLO
 ↓
Face Detection
 ↓
Face Embedding
 ↓
Local Similarity Search
 ↓
임시 인식 결과
```

Edge에서는 다음을 우선적으로 수행한다.

* 영상 전처리
* Object Detection
* Face Detection
* Face Embedding
* 기본적인 Face Matching
* 실시간 출석 판단

---

## Server

상대적으로 연산량이 크고 비실시간인 작업을 담당한다.

```text
Edge
 ↓
Embedding + Metadata
 ↓
Server
 ↓
Vector DB
 ↓
Active Learning
 ↓
Human Labeling
 ↓
Metric Learning
 ↓
Model Evaluation
 ↓
Model Update
```

그리고 필요한 경우:

```text
Vector Retrieval
       ↓
     RAG
       ↓
    LLM/VLM
       ↓
모델 성능 분석 / 데이터 분석
```

을 수행한다.

---

# 8. 권장 Edge-Server 구조

```text
                 ┌──────────────┐
                 │     CCTV     │
                 └──────┬───────┘
                        ▼
                ┌───────────────┐
                │   Edge AI     │
                │ YOLO           │
                │ Face Encoder   │
                └───────┬───────┘
                        │
                 Embedding +
                  Metadata
                        │
                        ▼
              ┌──────────────────┐
              │      Server      │
              │                  │
              │ Relational DB    │
              │ Vector DB        │
              │ Retrieval        │
              └────────┬─────────┘
                       │
                       ▼
              Active Learning
                       │
                       ▼
                Human Labeling
                       │
                       ▼
               Metric Learning
                       │
                       ▼
                 Model Update
                       │
                       ▼
                  Edge Deploy
```

이 구조를 통해 **실시간 추론은 Edge에서 수행하면서 모델 개선은 서버에서 수행하는 구조**를 구현할 수 있다.

---

# 9. RAG의 활용 영역

RAG는 실시간 얼굴 인식 자체보다 **운영 및 모델 개선 분석 영역**에서 활용하는 것을 권장한다.

예를 들어 관리자가 다음과 같이 질의할 수 있다.

> "최근 일주일간 얼굴 인식 실패가 가장 많이 발생한 조건은?"

RAG가 다음 데이터를 검색한다.

```text
Recognition Log
+
Face Embedding
+
Camera Metadata
+
Pose
+
Image Quality
+
Human Label
```

LLM/VLM이 이를 분석하여:

> "CAM-03에서 촬영된 우측 측면 얼굴과 저조도 환경에서 오인식률이 높으며, 김철수와 김영수 간의 Embedding 유사도가 높은 샘플이 반복적으로 확인되었습니다."

와 같은 분석을 제공할 수 있다.

---

# 10. 기대 효과

### ① 데이터 라벨링 비용 감소

전체 CCTV 데이터를 사람이 라벨링하지 않고 **학습 가치가 높은 데이터만 선별**할 수 있다.

### ② 얼굴 인식 성능 향상

Hard Positive / Hard Negative를 Metric Learning에 활용하여 유사 인물 간 오인식 문제를 개선할 수 있다.

### ③ 다양한 촬영 환경 대응

측면, 저조도, 가림, 원거리 등 실제 CCTV에서 발생하는 환경별 취약점을 지속적으로 보완할 수 있다.

### ④ 지속적인 모델 개선

```text
추론
 ↓
데이터 축적
 ↓
Hard Sample 검색
 ↓
Labeling
 ↓
Metric Learning
 ↓
모델 업데이트
 ↓
Edge 재배포
```

의 반복적인 **Continuous Learning Pipeline**을 구축할 수 있다.

---

# 11. 결론

본 시스템에서 기술별 역할은 다음과 같이 정의하는 것이 가장 적절하다.

| 기술                      | 주요 역할                             |
| ----------------------- | --------------------------------- |
| **YOLO / Vision Model** | 사람 및 얼굴 검출                        |
| **Face Encoder**        | 얼굴 → Embedding 변환                 |
| **Metric Learning**     | 얼굴 Embedding의 거리 관계 학습            |
| **Vector DB**           | 유사 얼굴 및 학습 데이터 검색                 |
| **Retrieval**           | Hard Positive/Negative 및 유사 사례 탐색 |
| **Active Learning**     | 학습 가치가 높은 데이터 선별                  |
| **RDB**                 | 학생·수업·출석·카메라 등 정형 데이터 관리          |
| **RAG + LLM/VLM**       | 검색 데이터를 기반으로 모델/데이터 분석 및 자연어 질의   |
| **Edge AI**             | 실시간 영상 추론 및 출석 처리                 |
| **Server**              | 데이터 관리, 학습, 모델 평가 및 업데이트          |

따라서 최종적으로는

> **Edge AI에서 실시간 얼굴 Embedding을 생성하고 → Vector DB에서 유사 사례를 검색하고 → Active Learning으로 Hard Sample을 선별하고 → Metric Learning으로 모델을 개선하고 → 개선된 모델을 다시 Edge에 배포하는 순환 구조**

를 핵심 아키텍처로 제안한다.

RAG는 이 구조의 **검색 결과와 DB/영상 Metadata를 LLM/VLM과 연결하여 "왜 모델이 실패했는가", "어떤 데이터를 추가 학습해야 하는가"를 분석하는 지능형 관리 계층**으로 활용하는 것이 적절하다.
