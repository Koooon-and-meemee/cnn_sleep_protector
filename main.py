import cv2
import numpy as np
from keras.models import load_model
import pygame
import time


def preprocess_img(img):
    # 이미지 크기 조정
    img = cv2.resize(img, (32, 32))
    # 이미지 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지 정규화
    gray = gray / 255.0
    # 모델 입력 형태에 맞게 차원 추가
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=-1)
    return gray

# 모델 불러오기
model = load_model('image_classification_model.h5')

# pygame 초기화
pygame.mixer.init()

# 기상 곡 파일 경로 설정
alarm_sound_path = '군대 기상나팔.mp3'

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

# 이전 프레임의 예측값 저장을 위한 변수
prev_prediction = None

while input("Press'q' to quir: ") != 'q':
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 이미지 전처리
    preprocessed_img = preprocess_img(frame)
    
    # 예측
    prediction = model.predict(preprocessed_img)

    if prev_prediction != 'he is alive' and prediction == 'he is alive':
        pygame.mixer.music.stop()
        time.sleep(2)
        continue

    if prev_prediction != 'he is alive' and prediction != 'he is alive':
        time.sleep(2)
        continue
    
    if prediction != 'he is alive':
        # 기상 곡 재생
        pygame.mixer.music.load(alarm_sound_path)
        pygame.mixer.music.play()
        prev_prediction = prediction
        time.sleep(2)
        continue


    # 잠시 대기
    time.sleep(2)

# 웹캠 캡처 종료
cap.release()
cv2.destroyAllWindows()
