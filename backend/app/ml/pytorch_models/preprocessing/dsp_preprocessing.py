import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def create_foreground_mask(img_rgb):
    """
    Generate a binary foreground mask using HSV information and morphology.
    Returns the binary mask (0 or 255, uint8).
    """
    # 1. Convert to HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # 2. Gaussian filtering to reduce noise
    s_channel = hsv[:, :, 1]
    blurred_s = cv2.GaussianBlur(s_channel, (5, 5), 0)
    
    # 3. Automatic thresholding using Otsu's method
    _, mask = cv2.threshold(blurred_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Clean mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Opening to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Closing to fill small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. Remove very small isolated regions (keep only the largest connected component)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned_mask = np.zeros_like(mask)
        cleaned_mask[labels == largest_label] = 255
        mask = cleaned_mask

    return mask

def apply_dsp_preprocessing(pil_image):
    """
    Applies DSP preprocessing: masking, background removal, resize, crop.
    Returns a PIL Image (RGB, 224x224).
    """
    img_rgb = np.array(pil_image.convert('RGB'))
    
    mask = create_foreground_mask(img_rgb)
    mask_3d = np.expand_dims(mask, axis=-1)
    fg_isolated = np.where(mask_3d == 255, img_rgb, 0).astype(np.uint8)
    
    fg_pil = Image.fromarray(fg_isolated)
    
    w, h = fg_pil.size
    target_size = 224
    
    if w < h:
        new_w = target_size
        new_h = int(target_size * h / w)
    else:
        new_h = target_size
        new_w = int(target_size * w / h)
        
    fg_pil = fg_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    left = (new_w - target_size) / 2
    top = (new_h - target_size) / 2
    right = (new_w + target_size) / 2
    bottom = (new_h + target_size) / 2
    
    fg_pil = fg_pil.crop((left, top, right, bottom))
    return fg_pil

class DSPTransform:
    def __call__(self, img):
        return apply_dsp_preprocessing(img)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_model1_dsp_pipeline(mean=None, std=None):
    """
    Returns the deterministic DSP preprocessing pipeline for Model 1.
    Original Image -> DSP (Mask+Resize) -> ToTensor -> Normalize
    """
    transform_list = [
        DSPTransform(),
        transforms.ToTensor()
    ]
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)
