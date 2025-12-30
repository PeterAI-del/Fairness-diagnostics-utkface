import os
import pandas as pd

def parse_utkface_metadata(image_path):
    fname = os.path.basename(image_path)
    try:
        age, gender, race, _ = fname.split('_')
        return int(age), int(gender), int(race), image_path
    except:
        return None

def load_utkface_dataset(root):
    data = []
    for img in os.listdir(root):
        entry = parse_utkface_metadata(os.path.join(root, img))
        if entry:
            data.append(entry)

    df = pd.DataFrame(data, columns=['age', 'gender', 'race', 'path'])
    return df

df = load_utkface_dataset("UTKFace/")


print("Dataset size:", len(df))
print(df['race'].value_counts())
print(df['gender'].value_counts())
print(df['age'].describe())


import tensorflow as tf
import os
import pandas as pd

# --- Dataset Preparation ---
df["gender"] = df["gender"].astype(int)

paths = df["path"].tolist()
labels = df["gender"].tolist()

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)     # safer than decode_image
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.shuffle(len(paths))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_size = int(0.8 * len(paths))

train_ds = (
    dataset.take(train_size)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    dataset.skip(train_size)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


# Upgrade to a new model
base = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  # freeze for speed

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)


# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=10)


# Plot ROC curve and AUC
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Collect true labels & predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    probs = model.predict(images)
    y_pred.extend(probs.ravel())
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()



