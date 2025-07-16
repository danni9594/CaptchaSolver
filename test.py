import os
import argparse
from captcha_solver import Captcha

def test_captcha_model(model_path, test_images):
    """Test the Captcha solver using the Captcha class."""
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        print("Please run train.py first to train the model.")
        return
        
    # Initialize the Captcha solver
    solver = Captcha(model_path)
    print(f"✓ Loaded model from {model_path}")
    
    # Test each image
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Define the output path
        result_filename = os.path.basename(img_path).replace('.jpg', '_predicted.txt')
        
        # Use the solver to predict
        solver(img_path, result_filename)

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
