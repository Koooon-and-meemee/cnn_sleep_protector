import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import plotext as plot
import time
from keras.models import load_model
import pygame



def load_data(dataset_path):
    images = []
    labels = []
    classes = os.listdir(dataset_path)
    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (32, 32))
            images.append(image)
            labels.append(i)
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    return images, labels

def model_training(dataset_path, model_name):
  images, labels = load_data(dataset_path)

  datagen = ImageDataGenerator(
      rotation_range = 1,
      width_shift_range=0.2,
      height_shift_range = 0.2,
      horizontal_flip = False
  )
  augmented_images = []
  augmented_labels = []

  for i in range(images.shape[0]):
      image = images[i]
      image = np.expand_dims(image, axis=-1)
      image = np.expand_dims(image, axis=0)
      label = labels[i]
      for _ in range(5):  # 각 이미지당 10장의 추가 생성
          for x_augmented, y_augmented in datagen.flow(image, [label], batch_size=1):
              augmented_images.append(np.squeeze(x_augmented))
              augmented_labels.append(y_augmented[0])
              break

  augmented_images = np.array(augmented_images)
  augmented_labels = np.array(augmented_labels)

  # 데이터 분할
  x_train, x_test, y_train, y_test = train_test_split(augmented_images, augmented_labels, test_size=0.2, random_state=42)

  # CNN 모델 정의
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(3, activation='softmax')  # 클래스 수만큼의 출력
  ])

  model.summary()

  # 모델 컴파일
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  # 모델 학습
  history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

  model_name = "C:\\Users\\User\\sleep_detector\\weights\\" + model_name +".h5" ##바꿔줘야해요

  model.save(model_name)

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

  accuracy_values = history.history['accuracy']
  val_accuracy_values = history.history['val_accuracy']
  epochs = range(1, len(accuracy_values) + 1)

  # 그래프 출력
  plot.plot(epochs, accuracy_values, label='Train Accuracy')
  plot.plot(epochs, val_accuracy_values, label='Validation Accuracy')
  plot.title('Model accuracy')
  plot.xlabel('Epoch')
  plot.ylabel('Accuracy')
  plot.grid(True)
  plot.show()

  print("The model training is complete.")
  return

def find_h5_files(directory):
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    return h5_files

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

def camera_logic(model_addr):

  model = load_model(model_addr)
  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  prev_prediction = 2  # 반복문 이전에 변수를 선언하고 초기화합니다.

  while True:
        ret, frame = capture.read()

        preprocessed_img = preprocess_img(frame)

        # 예측
        prediction = model.predict(preprocessed_img)
        prediction = np.argmax(prediction[0])

        if prev_prediction != 2 and prediction == 2:
            pygame.mixer.music.stop()
            print("music stop")
            time.sleep(1)
            prev_prediction = prediction
            continue
        
        if prev_prediction == 2 and prediction != 2:
            pygame.mixer.music.play()
            print("music start")
            time.sleep(1)
            prev_prediction = prediction
            continue
        
        
        time.sleep(1)
        prev_prediction = prediction  # 각 반복에서 이전 예측값을 업데이트합니다.



pygame.init()
pygame.mixer.init() 

current_directory = "C:\\Users\\User\\sleep_detector\\weights\\" ##파일 저장할 위치

h5_files_in_current_directory = find_h5_files(current_directory)

if h5_files_in_current_directory == []:
  print("Yon don't have any model")
  print("You need to train a new one")
  user_input = input("Enter Your dataset address and model name :")
  file_add, model_naame = user_input.split(",")
  model_training(file_add, model_naame)

h5_files_in_current_directory = find_h5_files(current_directory)

print("\n")
print("List of currently owned models")
print("=============================================")
for file_path in h5_files_in_current_directory:
  print(file_path)
print("=============================================")
new_one = 'o'
while(new_one == 'y' or 'n'):
    new_one = input("Do you want to train new model? ['y' or 'n']:")
    print('\n')
    if (new_one == 'y'):
       user_input = input("Enter Your dataset address and model name :")
       file_add, model_naame = user_input.split(",")
       model_training(file_add, model_naame)
       h5_files_in_current_directory = find_h5_files(current_directory)
       for file_path in h5_files_in_current_directory:
        print(file_path)
    elif (new_one == 'n'):
        n = input("Please enter the number of the model you want to run :")

        music_add = input("Please enter the address of your music file :")

        model_add = h5_files_in_current_directory[int(n)-1]
        original_path = model_add
        converted_path = original_path.replace("\\", "/")

        music_add = music_add.replace("\\", "/")

        pygame.mixer.music.load(music_add)
        camera_logic(converted_path)
