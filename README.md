# Captcha Solver

This project implements a ResNet-style Convolutional Neural Network (CNN) to solve 5-character captchas. The model is trained to recognize individual characters, which are then combined to predict the full captcha string.

## Methodology

### 1. Training and Prediction

The overall approach involves these key steps:
- **Image Cropping**: Each input captcha image is cropped into five smaller images, with each containing a single character.
- **Character Prediction**: The CNN model predicts the character in each of the five cropped images.
- **Training Data**: The training set consists of 25 captcha images, resulting in `25 * 5 = 125` individual character samples.
- **Prediction Target**: The model is used to predict the characters in `input100.jpg`.

### 2. Preprocessing

To ensure the model receives clean and consistent data, the following preprocessing is applied:
- **Fixed Bounding Box Trim**: A fixed number of pixels are trimmed from the sides of each captcha image. This ensures that the five equal-width character crops accurately capture each character without including edge artifacts.

### 3. Model Architecture

The model is a ResNet-style CNN, which includes:
- **Residual Blocks**: These blocks help prevent the vanishing gradient problem and allow for deeper, more effective networks.
- **Batch Normalization**: Stabilizes and accelerates the training process.
- **Adaptive Average Pooling and Dropout**: Reduces the risk of overfitting.

### 4. Training Procedure

The model is trained using the following configuration:
- **Optimizer**: AdamW (Adam with decoupled weight decay)
- **Learning Rate (lr)**: `1e-3` (default, configurable via command line)
- **Weight Decay**: `1e-4`
- **Batch Size**: `8` (default, configurable via command line)
- **Epochs**: `50` (default, configurable via command line)

## Usage

### 1. Environment Setup

To set up the required environment, follow these steps:

1. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   *Note: On Windows, the activation command is `.venv\Scripts\activate`*

2. **Install Required Packages**:
   Install the required packages using the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

### 2. Running the Project

- **To train the model**:
  ```bash
  python train.py --lr 0.001 --batch_size 8 --epochs 50
  ```
  You can customize the `lr`, `batch_size`, and `epochs` as needed.

  **Sample Output:**
  ```
  Using device: cpu
  Total training samples: 125
  Training samples: 125
  Starting training...
  Epoch [1/50], Loss: 3.6890, Accuracy: 2.40%
  Epoch [2/50], Loss: 3.2340, Accuracy: 17.60%
  Epoch [3/50], Loss: 2.7481, Accuracy: 32.80%
  ...
  Epoch [48/50], Loss: 0.0174, Accuracy: 100.00%
  Epoch [49/50], Loss: 0.0574, Accuracy: 99.20%
  Epoch [50/50], Loss: 0.0612, Accuracy: 98.40%
  Training completed. Model saved as 'captcha_model.pth'
  ```

- **To test the model**:
  Provide a list of simplified image names using the `--images` flag. If omitted, it defaults to testing `input100`.
  ```bash
  # Test specific images
  python test.py --images input00 input100

  # Test the default image (input100)
  python test.py
  ```

  **Sample Output (for `python test.py --images input00 input100`):**
  ```
  Testing ResNet-style CNN Captcha Solver
  ==================================================
  Using device: cpu
  ✓ Loaded model from captcha_model.pth

  Testing: input/input00.jpg
  Character 1: E (confidence: 0.998)
  Character 2: G (confidence: 0.999)
  Character 3: Y (confidence: 0.999)
  Character 4: K (confidence: 1.000)
  Character 5: 4 (confidence: 1.000)
  Predicted: EGYK4
  Result saved to: input00_predicted.txt

  Testing: input/input100.jpg
  Character 1: X (confidence: 0.473)
  Character 2: M (confidence: 0.973)
  Character 3: B (confidence: 0.991)
  Character 4: 1 (confidence: 0.660)
  Character 5: Q (confidence: 0.847)
  Predicted: XMB1Q
  Result saved to: input100_predicted.txt
  ```

### 3. Output Files

- **Saved Model**: After training, the model's weights are saved to `captcha_model.pth`.
- **Prediction Result**: When `test.py` is run, the predicted text for `input100.jpg` is saved to `input100_predicted.txt`.
