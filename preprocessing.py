import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import img_to_array, load_img
from keras.applications.efficientnet import preprocess_input
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# --- 1. SETTINGS ---
DATA_DIR = "data/raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# --- 2. OFFLINE AUGMENTATION (Saves physical files to disk) ---
def augment_class(class_dir, target_count=300):
    """Generates physical JPG files until the folder reaches target_count."""
    # Get list of original images
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)
    
    if current_count >= target_count:
        print(f"Skipping {class_dir}: Already has {current_count} images.")
        return

    # Augmentation configuration for saving files
    aug = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    needed = target_count - current_count
    generated = 0
    
    print(f"Augmenting {class_dir}: {current_count} -> {target_count}...")
    
    # Infinite loop over images until we hit the target
    while current_count + generated < target_count:
        for img_file in images:
            if current_count + generated >= target_count:
                break
                
            img = load_img(os.path.join(class_dir, img_file))
            arr = img_to_array(img)
            arr = arr.reshape((1,) + arr.shape)
            
            # flow() generates one batch of augmented images
            for batch in aug.flow(arr, batch_size=1, save_to_dir=class_dir, 
                                   save_prefix='aug', save_format='jpg'):
                generated += 1
                break # Move to next original image

# Ensure these folder names match your data/raw/ subfolders exactly
classes = ['cardboard', 'glass', 'metal', 'plastic']
for cls in classes:
    path = os.path.join(DATA_DIR, cls)
    if os.path.exists(path):
        augment_class(path, target_count=300)
    else:
        print(f"Warning: Folder not found: {path}")

# --- 3. DATA LOADERS (Online Preprocessing for Training) ---

# Training Generator with 'Online' RAM augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Essential for EfficientNet
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # Automatically splits the 300 images per class
)

# Validation Generator (No augmentation, just preprocessing)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- 4. VERIFICATION ---
print(f"\nTraining images: {train_generator.samples}")
print(f"Validation images: {val_generator.samples}")

# Visualize a batch (Colors will look weird due to preprocess_input)
images, labels = next(train_generator)
plt.figure(figsize=(10, 10))
for i in range(min(9, BATCH_SIZE)):
    plt.subplot(3, 3, i+1)
    # Note: We don't use 1./255 here because preprocess_input handles it
    # Clipping for visualization purposes only
    sample_img = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
    plt.imshow(sample_img)
    plt.title(list(train_generator.class_indices.keys())[labels[i].argmax()])
    plt.axis('off')
plt.show()