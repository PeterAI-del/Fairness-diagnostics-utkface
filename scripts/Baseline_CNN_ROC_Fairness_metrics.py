import os
import pandas as pd

def parse_utkface_metadata(fname):
    try:
        age, gender, race, _ = fname.split("_")
        return int(age), int(gender), int(race)
    except:
        return None

def load_utkface_dataset(root):
    rows = []
    for f in os.listdir(root):
        meta = parse_utkface_metadata(f)
        if meta:
            age, gender, race = meta
            rows.append([age, gender, race, os.path.join(root, f)])
    return pd.DataFrame(rows, columns=["age", "gender", "race", "path"])

df = load_utkface_dataset("UTKFace/")
print("Loaded:", len(df))


# Prepae TF Dataset
import tensorflow as tf

paths = df["path"].tolist()
labels = df["gender"].tolist()

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(5000)

train_size = int(0.8 * len(paths))

train_ds = dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds   = dataset.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)



# Old Model with weak baseline
old_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128,128,3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

old_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history_old = old_model.fit(train_ds, validation_data=val_ds, epochs=10)


# Get Prediction for Old Model
import numpy as np
from sklearn.metrics import roc_curve, auc

y_true_old = []
y_pred_old = []

for X, y in val_ds:
    p = old_model.predict(X).ravel()
    y_pred_old.extend(p)
    y_true_old.extend(y.numpy())

y_true_old = np.array(y_true_old)
y_pred_old = np.array(y_pred_old)



# Plot Old Model ROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

fpr, tpr, th = roc_curve(y_true_old, y_pred_old)
roc_auc_old = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"Old Model AUC = {roc_auc_old:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve — Old Model")
plt.legend()
plt.grid()
plt.show()

