# 🐾 Cats vs Dogs Image Classification with TensorFlow

This project builds a convolutional neural network (CNN) using TensorFlow and Keras to classify images of cats and dogs. It features a full pipeline from dataset preprocessing to model training, early stopping based on validation accuracy, and final predictions on test images.

## 📦 Dataset

- Source: [Cats and Dogs Filtered](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
- Automatically downloaded and extracted to: `~/.keras/datasets/cats_and_dogs_filtered`
- Contains separate folders for training images of cats and dogs.

## 🧠 Model Architecture

A Sequential CNN consisting of:

- 4 Convolutional layers with ReLU activation and MaxPooling
- Flattening layer for transition to dense layers
- Multiple fully connected Dense layers with:
  - Batch Normalization
  - Dropout regularization
- Output layer with Sigmoid activation for binary classification

## ⚙️ Training Strategy

- Uses `image_dataset_from_directory` with a 10% validation split.
- Trained using `model.fit()` for up to 100 epochs.
- Implements **custom early stopping**:
  - Stops training when validation accuracy doesn't improve for 5 consecutive epochs.
  - Automatically restores best weights after stopping.

## 📊 Visualization

Plots the training and validation accuracy/loss over time using Matplotlib:

```python
history_df[['loss', 'val_loss']].plot()
history_df[['accuracy', 'val_accuracy']].plot()
```

## 🐾 Prediction Function

Includes a custom prediction function that loads and preprocesses an image, runs inference, and outputs whether the image is classified as a cat or a dog:

```python
def predict_image(image_path):
    img = load_img(image_path, target_size=(200, 200))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    print("Dog 🐶" if result >= 0.5 else "Cat 🐱")
```

## 🚀 Highlights

- Stops training automatically at peak validation accuracy.
- Full training history recorded and visualized.
- Project structure allows easy scalability or transfer learning.

---

Built using 💙 TensorFlow and Python. Ideal for anyone learning image classification or experimenting with CNN optimization.
```
