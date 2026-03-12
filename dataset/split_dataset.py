import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем подготовленный CSV
df = pd.read_csv("dataset_labels.csv")

# Проверим баланс
print("Original distribution:")
print(df["label"].value_counts())

# Stratified split 80/20
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],  # сохраняем баланс классов
    random_state=42
)

# Проверяем распределение
print("\nTrain distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(val_df["label"].value_counts())

# Сохраняем
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)

print("\nSplit completed successfully.")