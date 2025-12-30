import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# 1. PARSE UTKFACE METADATA
# ------------------------------------

def parse_utkface_metadata(image_path):
    fname = os.path.basename(image_path)

    try:
        age, gender, race, _ = fname.split('_')
        return int(age), int(gender), int(race), image_path
    except:
        return None  # Skip files that don't parse correctly


def load_utkface_dataset(root):
    data = []
    for img in os.listdir(root):
        entry = parse_utkface_metadata(os.path.join(root, img))
        if entry:
            data.append(entry)

    return pd.DataFrame(data, columns=['age', 'gender', 'race', 'path'])


# ------------------------------------
# 2. LOAD FULL DATASET (ALL 23,705 IMAGES)
# ------------------------------------

df = load_utkface_dataset("UTKFace/")
print("Full dataset size:", len(df))
print(df.head())


# ------------------------------------
# 3. FULL DATASET — RACE DISTRIBUTION
# ------------------------------------

race_counts = df['race'].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(race_counts.index, race_counts.values)
plt.xlabel("Race label")
plt.ylabel("Count")
plt.title("UTKFace Race Distribution (Full Dataset)")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()


# ------------------------------------
# 4. FULL DATASET — GENDER DISTRIBUTION
# ------------------------------------

gender_counts = df['gender'].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(gender_counts.index, gender_counts.values)
plt.xlabel("Gender label (0 = Male, 1 = Female)")
plt.ylabel("Count")
plt.title("UTKFace Gender Distribution (Full Dataset)")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()


# ------------------------------------
# 5. FULL DATASET — AGE DISTRIBUTION
# ------------------------------------

plt.figure(figsize=(8,5))
plt.hist(df['age'], bins=40)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("UTKFace Age Distribution (Full Dataset)")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()

