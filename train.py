import tensorflow as tf
from tensorflow.keras import layers, models

# Correct path
dataset_path = "dataset_blood_group"

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(64, 64),
    batch_size=32
)

class_names = train_data.class_names
print("Classes:", class_names)

# Normalize
train_data = train_data.map(lambda x, y: (x/255.0, y))

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_data, epochs=5)

# Save
model.save("blood_model.h5")