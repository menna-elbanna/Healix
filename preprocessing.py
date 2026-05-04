import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

data_dir = "data/raw"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Classes:", train_generator.class_indices)
print("Train samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)

images, labels = next(train_generator)

plt.figure(figsize=(10, 6))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.axis('off')

plt.suptitle("Augmented Images")
plt.show()

import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os

data_dir = "data/raw"  
save_dir = "data/augmented" 

os.makedirs(save_dir, exist_ok=True)

IMG_SIZE = (224, 224)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

for class_name in os.listdir(data_dir):

    class_path = os.path.join(data_dir, class_name)
    save_class_path = os.path.join(save_dir, class_name)

    os.makedirs(save_class_path, exist_ok=True)

    generator = datagen.flow_from_directory(
        data_dir,
        classes=[class_name],
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode=None, # type: ignore
        save_to_dir=save_class_path,
        save_prefix='aug',
        save_format='jpg'
    )

    print(f"Processing {class_name}...")

    i = 0
    for batch in generator:
        i += 1
        if i > 150: 
            break