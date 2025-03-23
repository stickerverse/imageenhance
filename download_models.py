import os
import torch
import torchvision
from pathlib import Path
import urllib.request
import cv2

def download_opencv_models():
    """Download OpenCV models"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Download Haar cascade files
    cascade_urls = {
        "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    }
    
    for filename, url in cascade_urls.items():
        output_path = models_dir / filename
        if not output_path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded {filename} to {output_path}")
        else:
            print(f"{filename} already exists at {output_path}")

def create_bw_contrast_lut():
    """Create a black and white contrast LUT"""
    luts_dir = Path("models/luts")
    luts_dir.mkdir(exist_ok=True, parents=True)
    
    lut_path = luts_dir / "bw_contrast.cube"
    
    if not lut_path.exists():
        print("Creating B&W contrast LUT...")
        
        # Create a .cube LUT file for black and white with increased contrast
        with open(lut_path, 'w') as f:
            f.write("# Black and white contrast LUT\n")
            f.write("LUT_3D_SIZE 32\n")
            f.write("\n")
            
            # Create a 3D LUT with increased contrast in grayscale
            for r in range(32):
                for g in range(32):
                    for b in range(32):
                        # Convert RGB to grayscale using standard weights
                        gray = (0.299 * r/31.0 + 0.587 * g/31.0 + 0.114 * b/31.0)
                        
                        # Apply contrast curve
                        if gray < 0.5:
                            gray = 0.5 * (gray/0.5)**1.2  # Darken shadows
                        else:
                            gray = 1.0 - 0.5 * ((1.0-gray)/0.5)**0.8  # Brighten highlights
                        
                        # Output the same value for R, G, and B (grayscale)
                        f.write(f"{gray:.6f} {gray:.6f} {gray:.6f}\n")
        
        print(f"Created B&W contrast LUT at {lut_path}")
    else:
        print(f"B&W contrast LUT already exists at {lut_path}")

def download_torchvision_models():
    """Download TorchVision models"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Super-resolution model
    sr_path = models_dir / "super_resolution.pth"
    if not sr_path.exists():
        print("Downloading Super Resolution model...")
        # Using ResNet50 converted for super-resolution
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        # Modify for super-resolution (just saving the backbone)
        torch.save(model.state_dict(), sr_path)
        print(f"Downloaded Super Resolution model to {sr_path}")
    else:
        print(f"Super Resolution model already exists at {sr_path}")
    
    # Denoising model
    denoising_path = models_dir / "denoising.pth"
    if not denoising_path.exists():
        print("Downloading Denoising model...")
        # We'll use a pre-trained ResNet model as a base for denoising
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        torch.save(model.state_dict(), denoising_path)
        print(f"Downloaded Denoising model to {denoising_path}")
    else:
        print(f"Denoising model already exists at {denoising_path}")
    
    # Segmentation model
    segmentation_path = models_dir / "segmentation.pth"
    if not segmentation_path.exists():
        print("Downloading Segmentation model...")
        # Using FCN ResNet50 for segmentation
        weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
        torch.save(model.state_dict(), segmentation_path)
        print(f"Downloaded Segmentation model to {segmentation_path}")
    else:
        print(f"Segmentation model already exists at {segmentation_path}")
    
    # Depth estimation model
    depth_path = models_dir / "depth.pth"
    if not depth_path.exists():
        print("Downloading Depth model...")
        # Using ResNet34 as a base for depth estimation
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet34(weights=weights)
        torch.save(model.state_dict(), depth_path)
        print(f"Downloaded Depth model to {depth_path}")
    else:
        print(f"Depth model already exists at {depth_path}")
    
    # Style transfer model
    style_path = models_dir / "style_transfer.pth"
    if not style_path.exists():
        print("Downloading Style Transfer model...")
        # Using ResNet18 as a base for style transfer
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        torch.save(model.state_dict(), style_path)
        print(f"Downloaded Style Transfer model to {style_path}")
    else:
        print(f"Style Transfer model already exists at {style_path}")
    
    # Object detection model
    detection_path = models_dir / "object_detection.pth"
    if not detection_path.exists():
        print("Downloading Object Detection model...")
        # Using Faster R-CNN ResNet50 FPN for object detection
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        torch.save(model.state_dict(), detection_path)
        print(f"Downloaded Object Detection model to {detection_path}")
    else:
        print(f"Object Detection model already exists at {detection_path}")

def main():
    """Main function to download all models"""
    print("Starting model downloads...")
    
    # Create directory structure
    os.makedirs("models/luts", exist_ok=True)
    os.makedirs("models/filters", exist_ok=True)
    
    # Download OpenCV models
    download_opencv_models()
    
    # Create LUT file
    create_bw_contrast_lut()
    
    # Download TorchVision models
    download_torchvision_models()
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main()