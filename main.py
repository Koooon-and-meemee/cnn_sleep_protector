import cv2
import numpy as np
from keras.models import load_model
import time
import tensorflow as tf

model = load_model('much little filter with 200 epoch.h5')

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

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    preprocessed_img = preprocess_img(frame)


    
    # 예측
    prediction = model.predict(preprocessed_img)

    ##print(np.argmax(prediction))

    max_prob = np.max(prediction[0])

    # 첫 번째 인덱스 확률이 가장 높으면 1 출력, 아니면 0 출력
    if prediction[0][0] == max_prob:
        output = 1
    else:
        output = 0

    print(output)

    time.sleep(1)
    
