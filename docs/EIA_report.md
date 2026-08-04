조사해 오신 현장 데이터 분석 결과를 바탕으로 작성한 **최적의 데이터 전처리 파이프라인**입니다.

가장 눈여겨봐야 할 제약 조건은 **① 3m 이상 거리에서 FHD/15fps 이하 영상 처리**, **② 초저전력(NPU/MCU급) 엣지 디바이스**, **③ 플리커(Flicker) 현상 존재**입니다. 연산 자원이 극도로 제한된 환경이므로, 무거운 딥러닝 기반 전처리는 배제하고 **경량화된 수학적/픽셀 단위 전처리 파이프라인**을 설계했습니다.

---

## 🛑 핵심 현장 조건 분석 및 처리 전략

| 구분 | 현장 조건 | 전처리 전략 및 고려사항 |
| --- | --- | --- |
| **디바이스** | 초저전력 (NPU/MCU급) | **연산 최소화**: Deep Learning 기반 Super-Resolution, Heavy Alignment 완전 제외. C/C++(OpenCV) 기반 픽셀 연산만 사용. |
| **광원/품질** | 플리커(Flicker) 발생 | **주파수/시간축 디노이징**: 프레임 간 차이(Frame Difference) Smoothing으로 깜빡임 감쇄. |
| **카메라** | 3m+ 거리 / FHD / 2대 | **픽셀 해상도 부족**: 크롭된 얼굴 영역이 32x32 이하일 가능성 높음. 경량 Bicubic/Sharpening 필터 적용. |
| **네트워크** | Frame Crop 후 서버 전송 | **대역폭 최소화**: 엣지에서는 ROI 추출 및 BBox Filtering만 수행하고, 크롭된 픽셀 데이터만 서버로 전송. |

---

## 🔄 최적의 데이터 전처리 파이프라인

```
[입력 프레임 (FHD)]
       │
       ▼
 1. Frame Sampling & ROI Masking (Edge) ──► 불필요 연산 70% 감소
       │
       ▼
 2. Temporal Anti-Flicker Filter (Edge) ──► 플리커 노이즈 완화
       │
       ▼
 3. Lightweight Face Detection (Edge) ──► BBox 크기/Quality Filtering
       │
       ▼
 4. BBox Crop & Image Rescaling (Edge) ──► Unsharp Masking + Interpolation
       │
       ▼
 [중앙 서버 전송 (Crop 이미지)]
       │
       ▼
 5. Face Alignment & Normalization (Server) ──► 인식 알고리즘 투입

```

---

## 🛠️ 단계별 전처리 세부 구현 가이드

### 1. Frame Sampling & ROI Masking

* **목적**: 초저전력 엣지 보드의 FPS 저하 방지.
* **설정**: 15fps 중 초당 3~5프레임만 샘플링(Skip 3 frames). 강의실 내 좌석이 없는 벽/천장 영역은 Poly ROI Masking으로 즉시 연산 제외.

### 2. Temporal Anti-Flicker Filter

* **목적**: LED/형광등 주파수로 인한 프레임 간 밝기 진동 억제.
* **설정**: 복잡한 FFT 대신, 이전 N개 프레임의 Moving Average(이동 평균) 또는 Exponential Moving Average(EMA)를 적용하여 프레임 간 휘도 급변을 완화.

### 3. BBox Filtering & Quality Check

* **목적**: 인식 불가능한 저품질 Face Tile을 엣지 단계에서 미리 버려 서버 전송 및 연산 낭비 차단.
* **필터링 조건**:
* **크기 조건**: BBox Width or Height < 24px 제거
* **마스크/가림 조건**: Blur Metric(Laplacian Variance)이 기준치 미만인 경우 버림.



### 4. Image Rescaling & Edge Sharpening

* **목적**: 원거리 촬영으로 작아진 얼굴 영역 픽셀 강화.
* **설정**: `Bicubic Interpolation`으로 Target Size(예: 112x112)까지 확대한 후, 경량 `Unsharp Masking Filter`를 1회 적용해 이목구비 경계선을 명확히 보정.

---

## 💡 연산량 배분: Edge vs. Server

초저전력 MCU/NPU 디바이스 환경에 맞춘 역할 분담표입니다.

| 구분 | 엣지 디바이스 (MCU/NPU) | 중앙 서버 (Central Server) |
| --- | --- | --- |
| **역할** | 프레임 캡처, 플리커 필터, BBox Crop, 전송 | Face Landmark Alignment, Mask Detection, Re-ID, 출석 DB 기록 |
| **이유** | CPU/Memory 한계로 픽셀 단위 단순 연산만 수행 | Pose Alignment(3D) 및 heavy한 Feature Extraction 연산 수행 |

---

## 🐍 OpenCV Python 코드 스니펫

엣지 디바이스에서 동작 가능한 경량 전처리 파이프라인의 C/C++ 연동용 Python 참조 코드입니다.

```python
import cv2
import numpy as np

def preprocess_edge_frame(frame, roi_mask, prev_frame_gray=None, alpha=0.7):
    """
    초저전력 엣지 환경을 위한 경량 프레임 전처리
    """
    # 1. ROI Masking (연산 영역 제한)
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
    # 2. Frame Anti-Flicker (EMA 기반 단순 Temporal Smoothing)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    if prev_frame_gray is not None:
        smoothed_gray = cv2.addWeighted(gray, alpha, prev_frame_gray, 1 - alpha, 0)
    else:
        smoothed_gray = gray
        
    return smoothed_gray

def process_face_crop(face_crop, target_size=(112, 112)):
    """
    크롭된 원거리 작아진 얼굴 픽셀 보정
    """
    # 1. 크기 검증
    h, w = face_crop.shape[:2]
    if h < 24 or w < 24:
        return None  # 너무 작은 BBox 버림
        
    # 2. Sharpening Kernel (Unsharp Masking)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(face_crop, -1, kernel)
    
    # 3. Resizing (Bicubic)
    resized_face = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_CUBIC)
    
    return resized_face

```