import os
import cv2
import subprocess
import requests
import torch
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import supervision as sv
import os
import gdown
os.system('pip install gdown')

class YOLO_Detector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
    
    def detect(self, img_path):
        # Run YOLO detection and save output as txt
        label_name = os.path.basename(img_path).split('.')[0]
        results = self.model(img_path, save_txt=True, conf=0.30, save=True, show_labels=False)
        saved_path = results[0].save_dir
        # print(results[0].save_dir)
        # Read bounding box coordinates from txt file
        label_path = f"{saved_path}/labels/{label_name}.txt"

        # print(label_path)
        boxes = []
        image = cv2.imread(img_path)
        image_height, image_width = image.shape[:2]
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    _, x, y, w, h = map(float, line.strip().split())
                    # Convert YOLO format (cx, cy, width, height) to (x1, y1, x2, y2)
                    x1 = (x - w / 2) * image_width
                    y1 = (y - h / 2) * image_height
                    x2 = x1+ ( w * image_width)
                    y2 =  y1+ (h * image_height)
                    boxes.append([x1, y1, x2, y2])
        
        return np.array(boxes)

def segment_image(image_path, yolo_model_path, sam2_config_path, sam2_checkpoint_path):
    # Initialize YOLO detector
    detector = YOLO_Detector(model_path=yolo_model_path)
    boxes = detector.detect(img_path=image_path)

    # Setup CUDA and SAM2 model
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam2_model = build_sam2(sam2_config_path, sam2_checkpoint_path, device=DEVICE, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)

    # Prepare image for SAM2 and predict segmentation masks
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict(
        box=boxes,
        multimask_output=False
    )

    if boxes.shape[0] != 1:
        masks = np.squeeze(masks)

    # Annotate and plot segmented results
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks.astype(bool)
    )

    source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_images_grid(
        images=[source_image, segmented_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )
    return source_image,segmented_image


st.title("YOLO + SAM2 Segmentation Pipeline")
# Function to download files
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        f.write(response.content)
        
file_id = '1VFKc2sl-pGhdz5-FKQ8ZkajQRBeLPHQB'
yolo_model_path = "weights/Yolo/Yolov10m_best.pt"
sam2_config_path = "sam2_hiera_l.yaml"
sam2_checkpoint_path = "app/checkpoints/sam2_hiera_large.pt"
if not os.path.exists(sam2_checkpoint_path):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)
else:
    print("File already exists.")
    
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a local path
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Write the uploaded image to the local path
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Pass the path to segment_image
    source_image, segmented_image = segment_image(image_path, yolo_model_path, sam2_config_path, sam2_checkpoint_path)
    
    # Display the images in Streamlit
    st.image([source_image, segmented_image], caption=["Source Image", "Segmented Image"], width=300)
