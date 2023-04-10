from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw,ImageFilter
import numpy as np
import torch
import clip
import cv2

class SamClipDiffusion:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.image = None
        self.target = None
        self.prompt = None
        self.mask_generator = None
        self.masks = None
        self.cropped_boxes = None
        self.clip_model = None
        self.clip_preprocess = None
        self.probs = None
        self.image_path = None
        self.segmented_image = None
        
        
    def _get_mask_generator(self):
        # check if the weights are downloaded
        import os
        if not os.path.exists("sam_vit_h_4b8939.pth"):
            print("Downloading weights...")
            os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        
        mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
        self.mask_generator = mask_generator
        
    
    def _generate_masks(self, image_path):
        self.image_path = image_path
        self._get_mask_generator()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        masks = self.mask_generator.generate(image)
        self.masks = masks
        
    def _convert_box_xywh_to_xyxy(self, box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]
    
    def _segment_image(image, segmentation_mask):
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new("RGB", image.size, (0, 0, 0))
        transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
        transparency_mask[segmentation_mask] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image
    
    def _cut_all_masks(self,image_path):
        image = Image.open(image_path)
        
        cropped_boxes = []

        for mask in self.masks:
            cropped_boxes.append(self._segment_image(image, mask["segmentation"]).crop(self._convert_box_xywh_to_xyxy(mask["bbox"])))
        
        self.cropped_boxes = cropped_boxes     
        
    def _load_clip(self):
        device = self.device
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        self.clip_model = model
        self.clip_preprocess = preprocess
        
    @torch.no_grad()
    def _retrieve(self,elements: list[Image.Image], search_text: str) -> int:
        
        preprocessed_images = [self.clip_preprocess(image).to(self.device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(self.device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = self.clip_model.encode_image(stacked_images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100. * image_features @ text_features.T
        self.probs = probs[:, 0].softmax(dim=0)
        # return probs[:, 0].softmax(dim=0)
        
    def _get_indices_of_values_above_threshold(self, values, threshold):
        return [i for i, v in enumerate(values) if v > threshold]
    
    def segmentation_pipeline(self, image_path, target, threshold=0.05):
        self.image_path = image_path
        
        self._get_mask_generator()
        self._generate_masks(image_path)
        self._cut_all_masks(image_path)
        
        scores = self._retrieve(self.cropped_boxes, target)
        indices = self._get_indices_of_values_above_threshold(scores, threshold)

        segmentation_masks = []

        for seg_idx in indices:
            segmentation_mask_image = Image.fromarray(self.masks[seg_idx]["segmentation"].astype('uint8') * 255)
            segmentation_masks.append(segmentation_mask_image)

        original_image = Image.open(self.image_path)
        overlay_image = Image.new('RGBA', self.image.size, (0, 0, 0, 0))
        overlay_color = (255, 0, 0, 200)

        draw = ImageDraw.Draw(overlay_image)
        for segmentation_mask_image in segmentation_masks:
            draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

        result_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image)
        self.segmented_image = result_image
        return result_image
    
    