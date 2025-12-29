import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# ------------------------------------
# 1. LOAD UTKFACE METADATA
# ------------------------------------

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
    return pd.DataFrame(data, columns=['age', 'gender', 'race', 'path'])

df = load_utkface_dataset("UTKFace/")
print("Dataset size:", len(df))


# ------------------------------------
# 2. TF.DATA PIPELINE
# ------------------------------------

df["gender"] = df["gender"].astype(int)

paths = df["path"].tolist()
labels = df["gender"].tolist()

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Split
train_size = int(0.8 * len(paths))
train_ds = dataset.take(train_size).batch(32).shuffle(500).prefetch(tf.data.AUTOTUNE)
val_ds = dataset.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)

# Extract VAL metadata → IMPORTANT
val_df = df.iloc[train_size:].reset_index(drop=True)


# ------------------------------------
# 3. MODEL
# ------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)


# ------------------------------------
# 4. ROC + AUC
# ------------------------------------

y_true = []
y_pred = []

for images, labels in val_ds:
    probs = model.predict(images)
    y_pred.extend(probs.ravel())
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Gender Classification (Old Model)")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# ------------------------------------
# 5. BUILD RESULTS DATAFRAME — CORRECT VERSION
# ------------------------------------

results_old = val_df.copy()
results_old["y_true"] = y_true
results_old["y_pred_prob"] = y_pred
results_old["y_pred"] = (y_pred > 0.5).astype(int)


# ------------------------------------
# 6. DEMOGRAPHIC PLOTS (work now!)
# ------------------------------------

# --- Race Distribution ---
plt.figure(figsize=(8,5))
results_old['race'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Race label")
plt.ylabel("Count")
plt.title("UTKFace Race Distribution (Validation Subset)")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()

# --- Gender Distribution ---
plt.figure(figsize=(8,5))
results_old['gender'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Gender (0=Male, 1=Female)")
plt.ylabel("Count")
plt.title("UTKFace Gender Distribution (Validation Subset)")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()

# --- Age Distribution ---
plt.figure(figsize=(8,5))
plt.hist(results_old['age'], bins=40)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("UTKFace Age Distribution (Validation Subset)")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()


# ------------------------------------
# 7. FAIRNESS ANALYSIS
# ------------------------------------

print("\n=== Fairness by Gender ===")
for g in results_old["gender"].unique():
    group = results_old[results_old["gender"] == g]
    print(f"Gender {g}: ACC={accuracy_score(group['y_true'], group['y_pred']):.3f}, "
          f"AUC={roc_auc_score(group['y_true'], group['y_pred_prob']):.3f}")

print("\n=== Fairness by Race ===")
for r in results_old["race"].unique():
    group = results_old[results_old["race"] == r]
    print(f"Race {r}: ACC={accuracy_score(group['y_true'], group['y_pred']):.3f}, "
          f"AUC={roc_auc_score(group['y_true'], group['y_pred_prob']):.3f}")

print("\n=== Fairness by Age Group ===")
results_old["age_group"] = pd.cut(
    results_old["age"],
    bins=[0,20,40,60,80,120],
    labels=["0–20","21–40","41–60","61–80","80+"]
)

for ag in results_old["age_group"].unique():
    group = results_old[results_old["age_group"] == ag]
    print(f"Age {ag}: ACC={accuracy_score(group['y_true'], group['y_pred']):.3f}, "
          f"AUC={roc_auc_score(group['y_true'], group['y_pred_prob']):.3f}")
