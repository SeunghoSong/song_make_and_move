제시해주신 PRD 문서를 바탕으로, AI Vision 전문가 입장에서 ‘CCTV 기반 다중 안면 인식 시스템’이 목표 성능(15 FPS 이상, 98.5% 정확도)을 안정적으로 달성하기 위해 필요한 **전처리(Pre-processing)** 및 **후처리(Post-processing)** 핵심 기능을 정리해 드립니다.

---

## 1. 영상 및 이미지 전처리 (Pre-processing)

강의실 환경은 조명 변화, 멀리 앉은 학생의 정적/동적 가림, 광각 카메라의 왜곡 등 비정형 요소가 많습니다. 모델에 입력을 전달하기 전 엣지 AI(Jetson Orin Nano) 단에서 최적화하는 과정이 필수적입니다.

### ① 영상 스트림 수집 및 화질 정제

* **RTSP 버퍼링 및 프레임 드롭 제어**: CCTV RTSP 연결의 시차(Latency) 및 프레임 유실을 방지하기 위한 대기열(Queue) 관리.
* **렌즈 왜곡 보정 (Lens Distortion Correction)**: 광각 카메라 렌즈로 인해 외곽 영역 학생의 안면이 일그러지는 현상을 카메라 캘리브레이션 매트릭스($K, D$)로 보정.
* 
**ROI (Region of Interest) 마스킹**: 출입문, 창문, 출석 대상이 아닌 칠판/스크린 영역을 제외하고 **학생 좌석 구역만 ROI로 지정**하여 무의미한 연산 연쇄 차단.


* 
**조명/플리커(Flicker) 필터링**: 형광등 주파수 간섭이나 강의실 조명 변화에 대비해 CLAHE(Adaptive Histogram Equalization) 등으로 명암 대비 표준화.



### ② 안면 정밀 검출 및 표준화 (Detection & Alignment)

* 
**다중 안면 탐지 (Multi-Face Detection)**: YuNet / YOLOv8-Face를 사용하여 원거리 저해상도 안면 감지.


* 
**5-Point Landmark alignment**: 눈, 코, 입꼬리 위치를 기반으로 Affine Transformation을 적용하여 안면 정면 수평 축을 정렬 (ArcFace 입력 품질 극대화).


* **Data Augmentation 시뮬레이션 (초기 등록 단계)**:
* 단일 정면 프로필 사진을 3D Mesh 변환 후, Pitch(위/아래), Yaw(좌/우), Roll(회전) 다각도 임베딩으로 증강하여 Vector DB 저장.





---

## 2. 인식 결과 및 데이터 후처리 (Post-processing)

모델이 안면 특징 벡터(Embedding)를 추출한 후, 대리 출석을 방지하고 오인식을 차단하기 위한 통계적·네트워크적 검증 단계입니다.

### ① 특징점 검증 및 필터링 (Feature Level)

* 
**Face Quality Index (FQI) 평가**: Blurry(흔들림), 가림 비중(마스크, 모자 등), 안면 최소 크기(Pix size)가 기준 미달인 패치는 Recognition 단계로 넘기지 않고 기탁(Drop).


* **Liveness Detection (위변조 차단)**: 인쇄된 사진이나 스마트폰 화면을 카메라에 보여주는 부정 출석 방지 (Depth/Texture 기반의 Liveness 검증).

### ② 시간/공간 트래킹 및 Smoothing (Tracking Level)

* **Multi-Object Tracking (MOT)**: DeepSORT 또는 ByteTrack을 적용하여 수강생마다 고유 ID 부여.
* 프레임마다 안면을 매번 재인식하지 않고 '좌석에 앉아있는 동일인'으로 추적 유지.


* **Score Accumulation (시계열 매칭 통계)**: 단일 프레임 1회의 식별 결과로 출석을 즉시 확정하지 않고, **N개 프레임 동안 누적 코사인 유사도(Cosine Similarity) 평균값**이 Threshold(예: 0.75) 이상일 때 판정.

### ③ 2차 하이브리드 교차 검증 연동 (Dual Verification)

* 
**Wi-Fi BSSID & MAC/UUID Cross-Check**: Vision AI로 판정된 학생의 ID와 해당 시각 강의실 Wi-Fi AP(BSSID)에 접속된 모바일 앱 기기 정보 결합.


* 
**Dynamic QR 예외 파이프라인 전환**: FQI 미달 또는 유사도 미달로 최종 인식 실패(False Reject) 판정 시, 해당 학생에게 미인식 알림 발송 및 **TOTP Dynamic QR 인증 모드** 전환.



---

## 3. 전/후처리 파이프라인 전체 흐름도

$$\begin{array}{rcccl} \text{[CCTV RTSP Stream]} &\longrightarrow& \text{[1. 전처리: ROI/보정/Landmark]} &\longrightarrow& \text{[2. AI Model: Detection \& Extraction]} \\ &&&& \downarrow \\ \text{[출석 확정 (DB 저장)]} &\longleftarrow& \text{[4. 2차 Wi-Fi/QR 교차 검증]} &\longleftarrow& \text{[3. 후처리: FQI/MOT/누적 유사도]} \end{array}$$

* 
**전처리 핵심 핵심 목표**: 엣지 디바이스(Jetson Orin Nano)의 GPU 부하를 낮추고 **15 FPS 이상**을 유지할 수 있도록 정제된 Patch만 전달.


* 
**후처리 핵심 핵심 목표**: 조명/가림으로 인한 오탐을 스무딩(Smoothing) 기술로 보정하고 **안면 미인식률 2.0% 이하, 오인식률 0%** 수렴.