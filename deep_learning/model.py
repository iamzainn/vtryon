import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)

def free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class SegmentationModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def segment_person(self, image_path):
        try:
            # Load and preprocess the image
            original_image = Image.open(image_path).convert("RGB")
            image = original_image.copy()
            
            # Resize large images
            max_size = 1024  # You can adjust this value
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            
            input_tensor = self.preprocess(image).unsqueeze(0)
            
            try:
                input_tensor = input_tensor.to(self.device)
                self.model = self.model.to(self.device)
                
                # Perform segmentation
                with torch.no_grad():
                    output = self.model(input_tensor)['out'][0]
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU out of memory, falling back to CPU")
                    input_tensor = input_tensor.cpu()
                    self.model = self.model.cpu()
                    with torch.no_grad():
                        output = self.model(input_tensor)['out'][0]
                else:
                    raise e
            
            # Process the output
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
            # Class 15 corresponds to the 'person' class in PASCAL VOC dataset
            person_mask = (output_predictions == 15).astype(np.uint8) * 255
            
            # Apply morphological operations to refine the mask
            kernel = np.ones((5,5), np.uint8)
            person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
            person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours and keep only the largest one (assumed to be the person)
            contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                person_mask = np.zeros_like(person_mask)
                cv2.drawContours(person_mask, [largest_contour], 0, 255, -1)
            
            # Resize mask back to original image size
            person_mask = cv2.resize(person_mask, original_image.size[::-1], interpolation=cv2.INTER_NEAREST)
            
            return person_mask, original_image
        
        except Exception as e:
            print(f"An error occurred during segmentation: {str(e)}")
            return None, None
        
        finally:
            free_gpu_memory()

def visualize_segmentation(image, mask, output_path):
    # Convert PIL Image to numpy array if it's not already
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is RGB
    if image.shape[2] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Ensure mask and image have the same dimensions
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create RGB version of mask
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Ensure all arrays are uint8
    image = image.astype(np.uint8)
    mask_rgb = mask_rgb.astype(np.uint8)
    
    # Perform the blending
    result = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(result)
    plt.title('Segmentation Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# In the main block, add some debugging print statements:
if __name__ == "__main__":
    model = SegmentationModel()
    
    image_paths = [os.path.join(src_dir, "images", "man1.jpg"),
                   os.path.join(src_dir, "images", "man2.jpg"),
                   os.path.join(src_dir, "images", "woman.jpg")]
    
    for i, image_path in enumerate(image_paths):
        mask, original_image = model.segment_person(image_path)
        if mask is not None and original_image is not None:
            print(f"Image shape: {np.array(original_image).shape}")
            print(f"Mask shape: {mask.shape}")
            output_path = f'segmentation_result_{i+1}.png'
            visualize_segmentation(original_image, mask, output_path)
            print(f"Segmentation result saved as {output_path}")
        else:
            print(f"Segmentation failed for {image_path}")
        
       
        model.model = model.model.cpu()
        torch.cuda.empty_cache()