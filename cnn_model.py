import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터셋 폴더 경로
dataset_path = "path/to/dataset"

# 데이터 불러오기 및 전처리
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

images, labels = load_data(dataset_path)

# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

augmented_images = []
augmented_labels = []

for i in range(images.shape[0]):
    image = images[i]
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    label = labels[i]
    for _ in range(10):  # 각 이미지당 10장의 추가 생성
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
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# 모델 저장
model.save("image_classification_model.h5")
