import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from PIL import Image
import numpy as np

model = keras.Sequential()

train_data_dir = r"C:\Users\ricky\vsCodeWorkspace\tensorflow-test\galaxies\training"
verify_data_dir = r"C:\Users\ricky\vsCodeWorkspace\tensorflow-test\galaxies\verification"
num_classes = 3

# Define preprocessing and augmentation options
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

batch_size = 32
target_size = (224, 224)  # Change this to your desired target size

# Load and preprocess training data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and preprocess validation data
validation_generator = datagen.flow_from_directory(
    verify_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # num_classes is the number of classes in your problem
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

model.save("image_classification_model.h5")  # Save the entire model

loaded_model = load_model("image_classification_model.h5")

# Load and preprocess a new image
def preprocess_new_image(image_path):
    image = Image.open(image_path)
    image = image.resize(target_size)  # Resize to the target size used during training
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Preprocess new image and make a prediction
new_image = preprocess_new_image(r"C:\Users\ricky\vsCodeWorkspace\tensorflow-test\galaxies\testing\test-eliptical.jpg")
predictions = loaded_model.predict(new_image)

print(predictions)

class_names = ['eliptical', 'irregular', 'spiral']

# Get the index of the class with the highest predicted probability
predicted_class_index = np.argmax(predictions)

# Get the predicted class name
predicted_class_name = class_names[predicted_class_index]

# Print the predictions
print("Predicted class probabilities:", predictions)
print("Predicted class:", predicted_class_name)