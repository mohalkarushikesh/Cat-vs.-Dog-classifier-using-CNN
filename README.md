# 🐾 Cat vs. Dog Image Classifier

A simple Convolutional Neural Network (CNN) built with PyTorch to classify images of cats and dogs using the [Microsoft PetImages dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). Trained on a curated subset of 500 valid samples with early stopping for efficient performance.

---

## 🧠 Features

- CNN-based binary image classification (Cat 🐱 vs Dog 🐶)
- Handles corrupted images gracefully
- Uses only the first 500 valid samples for fast training
- Train/validation split using `scikit-learn`
- Early stopping based on validation accuracy
- Image prediction with live output

---

## 📁 Dataset Structure

Expected folder structure:

```
PetImages/
├── Cat/
│   └── *.jpg
├── Dog/
│   └── *.jpg
```

You can download the dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765) and place it in the project root.

---

## 🚀 Setup Instructions

1. Clone the repo and install dependencies:

```bash
pip install torch torchvision scikit-learn pillow
```

2. Download the PetImages dataset and unzip it in the project folder.

3. Run the training script:

```bash
python your_script_name.py
```

---

## 🔮 Predict New Images

To predict a new image using the trained model:

```python
# Load model
model = CatDogCNN().to(DEVICE)
model.load_state_dict(torch.load("catdog_model_light.pth"))
model.eval()

# Predict
predict_image("test_image.jpg", model)
```

Ensure `test_image.jpg` is a clean image of a cat or dog!

---

## 🛠 Architecture

```text
Input: RGB image resized to 150x150
→ Conv2D (32 filters) → ReLU → MaxPool
→ Conv2D (64 filters) → ReLU → MaxPool
→ Conv2D (128 filters) → ReLU → MaxPool
→ Flatten
→ Linear → ReLU
→ Output (2 classes: Cat, Dog)
```

---

## ✨ Future Ideas

- Add data augmentation (flip, rotate, shift)
- Use dropout to reduce overfitting
- Streamlit UI for live image uploads and predictions
- Training progress visualization with `matplotlib` or `wandb`

---

## 👤 Author

Built with 💻, 🧠, and a love for furry friends.

```
