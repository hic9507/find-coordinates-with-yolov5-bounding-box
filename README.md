# Yolov5를 이용해 획득한 이미지 상의 바운딩 박스를 이용해 원하는 좌표를 얻는 방법

### 1. Yolov5 학습을 통해 생성된 바운딩 박스가 그려진 이미지 준비
### 2. 바운딩 박스가 존재하지 않는 원본 이미지를 이진화 및 canny 등의 opencv 연산 적용
### 3. 바운딩 박스의 꼭지점 좌표가 주어지므로, 바운딩 박스가 없는 원본 이미지에서 바운딩 박스의 각 좌표로부터 일정 간격으로 픽셀을 이동
### 4. 픽셀을 이동했을 때, 해당 픽셀의 값이 1이면(이진화 등의 연산을 적용했으므로 픽셀 값이 객체는 1, 그 외는 0이 됨)
### 5. 픽셀을 이동하며 픽셀의 값이 1인 부분을 노랗게 표현
### 6. 노란 부분이 제일 많은 지점의 픽셀이 실제 신분증의 테두리 부분
### 7. 해당 노란 테두리를 바탕으로 다시 테두리를 그림(contour 사용)
### 8. 신분증 테두리 검출

### 예시 화면
![예시](https://github.com/hic9507/find-coordinates-with-yolov5-bounding-box/assets/65755028/6362a457-119d-43d0-8b38-cf7249565012)


## line_detector.py
가우시안 블러 등의 연산을 수행하고 픽셀을 순회하며 검출 및 노란 라인을 그리는 코드

## move.py
실제 실행하는 코드로, 객체 밖의 좌표가 검출되지 않게 하는 역할을 하며 line_detector.py가 계산한 좌표를 저장함.

픽셀 이동 범위는 지정 가능함.

### 실행 화면
![실행화면](https://github.com/hic9507/find-coordinates-with-yolov5-bounding-box/assets/65755028/e0581c47-3498-4eef-8647-f9af09293ab2)

### 결과 화면
![결과 화면](https://github.com/hic9507/find-coordinates-with-yolov5-bounding-box/assets/65755028/24132ca5-217c-40e7-af1d-521e77c28aac)

### 상세 시연 과정
![상세 시연과정 1](https://github.com/hic9507/find-coordinates-with-yolov5-bounding-box/assets/65755028/90a5ce6c-a426-4713-9ae1-2449b6e84201)
![상세 시연과정 2](https://github.com/hic9507/find-coordinates-with-yolov5-bounding-box/assets/65755028/ee06df29-d553-4f64-8b84-1bd378ab52d7)
