# File: train.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Paths
base_path = "C:/Users/damar/Downloads/aiproject/brain_tumor_data"
train_dir = os.path.join(base_path, "Training")
img_size = (150, 150)
batch_size = 32

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_data.class_indices)
print("Found classes:", train_data.class_indices)

# GA Parameters
GA_DROPOUT = [0.2, 0.3, 0.4, 0.5]
GA_DENSE = [64, 96, 128, 160, 192, 256]
GA_LR = [1e-3, 5e-4, 1e-4]

def create_individual():
    return {
        'dropout_rate': random.choice(GA_DROPOUT),
        'dense_units': random.choice(GA_DENSE),
        'learning_rate': random.choice(GA_LR)
    }

def build_model(dropout_rate=0.3, dense_units=128, learning_rate=1e-4):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_individual(individual):
    model = build_model(**individual)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=1, restore_best_weights=True)
    history = model.fit(train_data, epochs=3, validation_data=val_data, callbacks=[early_stop], verbose=0)
    return max(history.history['val_accuracy'])

# Genetic Algorithm
population = [create_individual() for _ in range(5)]
for gen in range(3):
    print(f"\nGeneration {gen + 1}")
    fitnesses = [evaluate_individual(ind) for ind in population]
    for i, (ind, acc) in enumerate(zip(population, fitnesses)):
        print(f"Individual {i + 1}: {ind}, Accuracy: {acc:.4f}")

    elite = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[0][0]
    parents = [x[0] for x in sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:2]]
    new_population = [elite]

    while len(new_population) < 5:
        p1, p2 = random.sample(parents, 2)
        child = {
            'dropout_rate': random.choice([p1['dropout_rate'], p2['dropout_rate']]),
            'dense_units': random.choice([p1['dense_units'], p2['dense_units']]),
            'learning_rate': random.choice([p1['learning_rate'], p2['learning_rate']])
        }
        if random.random() < 0.2:
            child['dropout_rate'] = random.choice(GA_DROPOUT)
        if random.random() < 0.2:
            child['dense_units'] = random.choice(GA_DENSE)
        if random.random() < 0.1:
            child['learning_rate'] = random.choice(GA_LR)
        new_population.append(child)

    population = new_population

# Final Model Training
final_model = build_model(dropout_rate=0.4, dense_units=256)

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
checkpoint = ModelCheckpoint("brain_tumor_model_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

history = final_model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Fine-tuning
print("Fine-tuning last 20 layers...")
final_model.trainable = True
for layer in final_model.layers[:-20]:
    layer.trainable = False

final_model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_history = final_model.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Save Final Model
final_model.save("brain_tumor_model_final.h5")
print("✅ Final model saved as brain_tumor_model_final.h5")

# Plot Training Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + fine_tune_history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'] + fine_tune_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()