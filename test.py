import os
import torch
from PIL import Image
from captcha_solver import CaptchaCNN, CHARS, trim_fixed
import argparse

def test_captcha_model(model_path, test_images):
    """Test the ResNet-style CNN model"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = CaptchaCNN(num_classes=36)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"✗ Model file not found: {model_path}")
        print("Please run train.py first to train the model.")
        return
    
    model.to(device)
    model.eval()
    
    # Transform for preprocessing
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Test each image
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\nTesting: {img_path}")
        
        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        
        # Apply the same fixed trim as used in training
        image = trim_fixed(image)
        
        # Predict each character
        predicted_chars = []
        char_width = image.width // 5
        
        with torch.no_grad():
            for i in range(5):
                # Extract character segment
                char_img = image.crop((i * char_width, 0, (i + 1) * char_width, image.height))
                
                # Convert to grayscale and resize to 32x32
                char_img = char_img.convert("L")
                # Use Image.LANCZOS for older Pillow versions
                char_img = char_img.resize((32, 32), Image.LANCZOS)
                
                # Convert to tensor
                char_tensor = transform(char_img).unsqueeze(0).to(device)
                
                # Predict
                output = model(char_tensor)
                pred_idx = output.argmax(dim=1).item()
                predicted_char = CHARS[pred_idx]
                predicted_chars.append(predicted_char)
                
                # Print confidence scores for debugging
                probs = torch.softmax(output, dim=1)
                confidence = probs.max().item()
                print(f"Character {i+1}: {predicted_char} (confidence: {confidence:.3f})")
        
        # Combine prediction
        predicted_captcha = ''.join(predicted_chars)
        print(f"Predicted: {predicted_captcha}")
        
        # Save result to file
        result_filename = os.path.basename(img_path).replace('.jpg', '_predicted.txt')
        with open(result_filename, 'w') as f:
            f.write(predicted_captcha + '\n')
        print(f"Result saved to: {result_filename}")

def main():
    parser = argparse.ArgumentParser(description='Test the Captcha-solving CNN model.')
    parser.add_argument('--images', nargs='+', default=['input100'], help='List of simplified image identifiers (e.g., input00, input100).')
    parser.add_argument('--model_path', default='captcha_model.pth', help='Path to the trained model file.')
    
    args = parser.parse_args()
    
    # Construct full image paths
    test_images = [f"input/{name}.jpg" for name in args.images]
    
    print("Testing ResNet-style CNN Captcha Solver")
    print("="*50)
    
    # Test the model
    test_captcha_model(args.model_path, test_images)

if __name__ == "__main__":
    main()
