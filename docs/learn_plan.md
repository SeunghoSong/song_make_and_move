AI Vision 전문가로서, 라즈베리파이(Raspberry Pi 4/5)와 같은 제한된 자원의 엣지(Edge) 기기 환경에서 PRD 요구사항(15 FPS 이상 실시간 처리, 다중 안면 탐지, 조명/가림 대응)을 만족하기 위한 **YOLO 모델 선정, Roboflow 데이터셋 구성, 학습 방법론**을 설명해 드립니다.

---

### 1. YOLO 모델 선정 및 이유

#### 💡 추천 모델: **YOLOv8n-Face** 또는 **YOLO11n-Face (Nano 버전)**

* **선정 이유**:
1. **엣지 기기 연산 한계 극복 (Ultra-lightweight)**
* 라즈베리 파이는 GPU 성능이 없거나 소형 NPU(예: Hailo-8L 등)에 의존해야 합니다.
* Nano(`n`) 모델은 파라미터 수가 약 3M 미만, FLOPs가 매우 적어 라즈베리 파이 5 환경에서도 CPU(ONNX/NCNN 경량화) 단독으로 **15~25 FPS** 이상을 확보할 수 있습니다.


2. **원거리 다중 안면(Crowded Face) 감지 능력**
* 일반 YOLOv8/11 기본 모델(COCO 80 클래스)과 달리, **Face 전용 Anchor/Head로 파인튜닝된 YOLO-Face 계열**은 강의실 멀리 앉아있는 소형 안면(Small Face) 탐지 성능이 압도적으로 우수합니다.


3. **경량 포맷(ONNX/NCNN/OpenVINO) 변환 최적화**
* PyTorch(`.pt`)로 학습 후 엣지 배포용 양자화(INT8 Quantization) 및 ONNX/TFLite 변환 호환성이 검증된 모범 표준 아키텍처입니다.


---

### 2. Roboflow 데이터셋 선정 및 이유

#### 💡 추천 데이터셋: **Roboflow Universe의 `WIDER FACE (YOLO Format)` + `Classroom Crowd Face` 커스텀 병합 데이터셋**

* **선정 이유**:
1. **강의실 비정형 환경 모사 (가림/조명/자세)**
* PRD 문서의 위험 요소로 지적된 "조명 변수, 마스크/모자/숙인 자세(Occlusion)"를 극복하려면 단순 정면 증명사진 데이터셋으로는 불가능합니다.
* WIDER FACE 데이터셋에는 Scales(크기), Poses(자세), Occlusions(가림), Illuminations(조명)에 대한 난이도별 안면 레이블이 포함되어 있어 강인한 탐지 성능을 보장합니다.


2. **다중 안면 밀집 데이터 (High Density)**
* 한 프레임 내 20~50명 이상이 동시에 캡처되는 강의실 특성에 맞게, 프레임당 안면 Bounding Box 레이블이 dense하게 채워진 Crowd 데이터셋이 필수적입니다.


3. **Data Augmentation(데이터 증강) 옵션 활용**
* Roboflow 플랫폼 자체 기능을 통해 Mosaic, Random Cutout, Brightness Adjust(-25% ~ +25%), Rotation(-15° ~ +15°)을 적용하여 PRD의 3D Pose/다각도 대응 비전 성능을 보완할 수 있습니다.

---

### 3. 데이터셋 기반 YOLO 모델 학습 방법 (Training Pipeline)

전체 학습 프로세스는 "데이터 정제 → 전이 학습 → 검증 → 엣지 경량화 포팅"의 4단계로 진행됩니다.

#### 1단계: Roboflow 데이터 정제 및 파이프라인 구축

* **Roboflow Export**:
* 이미지 해상도를 **`640x640`** (또는 라즈베리파이 속도 최우선 시 `416x416`)으로 Resize 처리하여 Export.
* 단일 클래스(`face`)로 레이블링 통합.

#### 2단계: 전이 학습 (Transfer Learning)

* **Pre-trained Weights 활용**:
* 처음부터 학습(Train from scratch)하지 않고, 미리 사전 학습된 `yolov8n-face.pt` 혹은 `yolo11n.pt` 가중치를 전이 학습(Fine-tuning)합니다.


* **주요 하이퍼파라미터 설정**:
* **Epochs**: 100 ~ 150 (Early Stopping `patience=15` 설정)
* **Batch Size**: 16 또는 32 (학습용 GPU PC 기준)
* **Image Size (`imgsz`)**: 640
* **Augmentation**: Mosaic=1.0, Mixup=0.15, Fliplr=0.5 (좌우 반전)

#### 3단계: 성능 검증 (Evaluation)

* **KPI 확인**: PRD 상의 KPI 목표인 **인식 정확도 98.5% 이상**, **미인식률(False Reject) 2% 이하**에 맞춰 테스트 세트의 **mAP50 및 mAP50-95** 지표 평가.
* mAP50 기준으로 최소 **0.95(95%) 이상** 확보해야 원거리 강의실 안면 탐지가 안정적으로 구동됩니다.

#### 4단계: 엣지(라즈베리파이) 배포용 경량화 (Optimization)

* **양자화(Quantization) 및 Export**:
```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
# 엣지용 ONNX 포맷 변환 (INT8 / FP16 양자화 적용)
model.export(format="onnx", int8=True, imgsz=640)

```
* **라즈베리파이 추론 엔진 적용**:
* 라즈베리파이 상에서는 Heavy한 PyTorch 대신 **ONNX Runtime** 또는 **NCNN / OpenVINO** 파이썬/C++ 런타임을 활용해 프레임 추론 속도를 2~3배 끌어올려 **15 FPS 이상** 목표를 달성합니다.