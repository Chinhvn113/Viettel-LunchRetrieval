import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Use automatic mask generator instead of predictor
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
predicter = SAM2ImagePredictor(sam2)

# Load and process image
image = Image.open('/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/0057.png')
image = np.array(image.convert("RGB"))
def predict(image, input_point, input_label):
    predicter.set_image(image)
    masks, scores, logits = predicter.generate_automatic_masks(point_coords=input_point, 
                                              point_labels=input_label,
                                              multimask_output=True)
    return masks, scores, logits
###predict and save image for visualization
