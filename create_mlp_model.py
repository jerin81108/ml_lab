import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Create a dummy model for character recognition (A-Z)
def create_and_save_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(26, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Save with the specific name the user's script looks for
    model.save('handwritten_character_recog_model.h5')
    print("Created and saved 'handwritten_character_recog_model.h5'!")

if __name__ == "__main__":
    create_and_save_model()
