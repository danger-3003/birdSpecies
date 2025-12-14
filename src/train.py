import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from dataset_loader import load_dataset
from models.crnn_model import build_crnn
from config import DATASET_PATH, MODEL_PATH, BATCH_SIZE


# -----------------------------
# Load dataset
# -----------------------------
print("🔄 Loading dataset...")
X, y, encoder = load_dataset(DATASET_PATH)

labels = np.argmax(y, axis=1)

# -----------------------------
# Detect rare classes
# -----------------------------
unique, counts = np.unique(labels, return_counts=True)
class_counts = dict(zip(unique, counts))

print("📊 Samples per class:")
for cls, cnt in class_counts.items():
    print(f"Class {cls} ({encoder.classes_[cls]}): {cnt}")

# Keep only classes with >= 2 samples
valid_classes = [cls for cls, cnt in class_counts.items() if cnt >= 2]

print("\n✅ Valid classes:", len(valid_classes))
print("❌ Removed classes:", len(class_counts) - len(valid_classes))

# Filter dataset
valid_indices = np.isin(labels, valid_classes)

X = X[valid_indices]
y = y[valid_indices]
labels = labels[valid_indices]

# -----------------------------
# Stratified split (SAFE NOW)
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Validation samples: {len(X_val)}")


# -----------------------------
# Build CRNN model
# -----------------------------
model = build_crnn(
    input_shape=X.shape[1:],
    num_classes=y.shape[1]
)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]


# -----------------------------
# Train model
# -----------------------------
print("🚀 Training started...")
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)


# -----------------------------
# Save model
# -----------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
