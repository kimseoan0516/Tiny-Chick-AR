# Tiny-Chick-AR

OpenCV를 활용하여 카메라의 자세(Pose)를 추정하고, 실시간 영상 속 체스보드 위를 생동감 있게 움직이는 **3D 병아리 AR 모델**을 투영하는 증강현실(AR) 어플리케이션입니다.

## 개요 (Overview)
사전 계산된 카메라 캘리브레이션 파라미터와 영상을 이용하여 3D-2D 투영 관계를 확립하고, 카메라의 6자유도 위치 및 회전을 실시간으로 획득합니다. 추정된 카메라 시점을 바탕으로, 역동적으로 걷는 3D 병아리 객체를 실제 환경(비디오 스트림) 위에 이질감 없이 병합하여 렌더링합니다.

## 주요 기능 (Features)

- **실시간 카메라 자세 추정 (Pose Estimation)**: `cv2.solvePnP`를 통해 프레임별 이동(Translation) 및 회전(Rotation) 벡터를 연산하여 카메라의 동선을 정밀하게 트래킹합니다.
- **병아리 모델 렌더링 (Custom 3D Engine)**: 
  - 기본 제공되는 단순 선분 예제가 아닌, 텍스처와 컬러가 적용된 **병아리 `.obj` 및 `.mtl` 모델 데이터**를 자체 엔진으로 파싱하고 시각화합니다.
  - **화가 알고리즘 (Painter's Algorithm)**: 병아리 모델의 다각형 면 깊이(Depth)를 계산해 멀리 있는 뒷면부터 앞면 순서로 그려 올바른 원근감을 구현.
  - **램버트 셰이딩 (Lambertian Shading)**: 광원 벡터에 기반하여 3D 병아리 곡면에 생기는 음영과 명암을 사실적으로 표현.
- **자연스러운 병아리 애니메이션**: 병아리가 한 곳에 정지해 있는 것이 아니라 가상의 체스보드 공간 위를 이리저리 이동하며, 이동 진행 방향에 맞추어 고개(시선)를 스스로 돌리도록 상호작용이 구현되어 있습니다.
- **공중 부양 텍스트 효과**: 공간 한가운데에 고정되어 렌더링되는 "AR DEMO" 네온 텍스트를 배치하였습니다.

## 실행 결과 및 데모 (Results & Demo)

### 📸 AR 렌더링 결과 이미지
![ar_demo_1](https://github.com/user-attachments/assets/5ea24a2d-293c-4ae5-a64f-96f2186b4305)
![ar_demo_2](https://github.com/user-attachments/assets/414100b3-82ab-4bec-aa63-ba5e2a3dd25b)


### 🎥 동작 시연 영상

https://github.com/user-attachments/assets/6105c804-6f66-4afa-94c8-1c1ecd8b825d

* 스크립트 실행 후 전체 동작 시연 비디오가 `results/ar_result.avi` 로컬 경로에 자동 저장됩니다.
* 프로그램 실행 중 주요 프레임의 시연 스크린샷 역시 자동 생성됩니다.

## 실행 방법 (How to Run)

### 요구 의존성 패키지
- Python 3.x
- `opencv-python`
- `numpy`

### 구동 절차
1. 카메라 캘리브레이션 데이터 (`calibration_result.pkl`) 및 모델 데이터(`.obj` 등), 영상 데이터가 알맞은 경로에 위치하는지 확인합니다.
2. 터미널(또는 명령 프롬프트)에서 아래의 명령어를 입력하여 프로그램을 실행합니다:
   ```bash
   python ar_pose_estimation.py
   ```
