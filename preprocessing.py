import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- 1. SETTINGS ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}. "
        "Expected structure: data/raw/<class_name>/*.jpg"
    )

# --- 2. THE AUGMENTATION ENGINE (The "RAM" Part) ---
# This creates infinite variations in memory while the script runs
train_datagen = ImageDataGenerator(
    rotation_range=30,      # Tilted images
    width_shift_range=0.2,  # Moved sideways
    height_shift_range=0.2, # Moved up/down
    zoom_range=0.2,         # Zoomed in/out
    horizontal_flip=True,   # Mirrored
    fill_mode='nearest',    # Fills in gaps from rotation
    validation_split=0.2    # Reserves 20% for testing
)

# --- 3. DATA LOADERS ---
# This pulls from 'data/raw' but applies the "twists" on the fly
train_generator = train_datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
# Fix: separate val datagen
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(  # use val_datagen, not train_datagen
    str(DATA_DIR), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)
# --- 4. VERIFICATION ---
print(f"\nSuccessfully loaded {train_generator.samples} training images.")
print(f"Successfully loaded {val_generator.samples} validation images.")

# Let's grab one batch to see the RAM magic
images, labels = next(train_generator)

# This shows the first 9 images from that memory-batch
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.title(list(train_generator.class_indices.keys())[labels[i].argmax()])
    plt.axis('off')

plt.suptitle("Augmented Images (Generated in RAM)")
plt.show()