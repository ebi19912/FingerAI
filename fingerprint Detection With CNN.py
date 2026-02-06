from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

# ایجاد مدل CNN
model = Sequential()
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

# ذخیره مدل تشخیص اثر انگشت
model.save('fingerprint_model.h5')

# بارگذاری مدل تشخیص اثر انگشت
model = load_model('fingerprint_model.h5')

# بارگذاری جدول مالکیت انگشت‌ها
fingerprint_ownership = {
    1: "Person A",
    2: "Person B",
    # ادامه ...
}

# بارگذاری و پیش‌پردازش تصویر اثر انگشت مورد آزمایش
test_image = image.load_img('test_fingerprint.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# پیش‌بینی با استفاده از مدل تشخیص اثر انگشت
result = model.predict(test_image)
if result[0][0] == 1:
    # اگر اثر انگشت تشخیص داده شود
    fingerprint_id = 1  # تعیین شماره انگشت
    if fingerprint_id in fingerprint_ownership:
        owner = fingerprint_ownership[fingerprint_id]
        print(f'This fingerprint belongs to {owner}.')
    else:
        print('Fingerprint owner not found.')
else:
    print('This is not a fingerprint.')
