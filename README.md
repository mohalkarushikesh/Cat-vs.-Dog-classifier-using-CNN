# 🐾 Cats vs Dogs Classification with CNN

This project demonstrates how to build a convolutional neural network (CNN) using TensorFlow and Keras to classify images of cats and dogs.

## 📦 Dataset

- Source: [Cats and Dogs Filtered Dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
- Structure:
  - `/train/cats/`
  - `/train/dogs/`
  - Images are extracted to: `~/.keras/datasets/cats_and_dogs_filtered`

## 🛠️ Setup

```bash
pip install tensorflow pandas numpy matplotlib
```

## 📁 Directory Split

Using `image_dataset_from_directory` to split training and validation data:

```python
image_dataset_from_directory(
    base_dir,
    image_size=(200, 200),
    validation_split=0.1,
    subset='training' / 'validation',
    seed=1,
    batch_size=32
)
```

## 🧠 Model Architecture

Sequential CNN with multiple convolution and pooling layers:

- Conv2D ➝ MaxPooling ➝ Flatten
- Dense ➝ BatchNormalization ➝ Dropout
- Final layer with `sigmoid` activation for binary classification

## ⚙️ Compilation & Training

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(train_data, epochs=10, validation_data=val_data)
```

## 📊 Evaluation

Training and validation accuracy/loss curves are plotted using Matplotlib.

## 🔍 Prediction

Function `predict_image(image_path)` loads and predicts if the input is a **cat** or a **dog**:

```python
img = image.load_img(image_path, target_size=(200, 200))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
model.predict(img_array)
```

## 📷 Sample Predictions

```python
predict_image('/path/to/cat.jpg')
predict_image('/path/to/dog.jpg')
```

## 🚀 Future Improvements

- Add data augmentation for robustness
- Use pretrained models like EfficientNet for higher accuracy
- Include EarlyStopping and LearningRateScheduler for smarter training

---

Built with 💙 using TensorFlow + Keras. Happy coding!
```
