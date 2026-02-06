from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# ایجاد مدل CNN
model = Sequential()

# اضافه کردن لایه‌ها
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# کامپایل کردن مدل
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# پیاده‌سازی ImageDataGenerator برای افزایش داده
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# آموزش مدل
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')
model.fit(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)


import numpy as np
from keras.preprocessing import image

# بارگذاری و پیش‌پردازش تصویر
test_image = image.load_img('single_prediction/test_image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# پیش‌بینی با استفاده از مدل
result = model.predict(test_image)
if result[0][0] == 1:
    print('This is a fingerprint.')
else:
    print('This is not a fingerprint.')
# کامپایل کردن مدل
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

