import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import clip
import torch
import numpy as np
class ImageSegmenter:
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", clip_model="ViT-B/32"):
        self.mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=sam_checkpoint))
        self.clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.clip_device)

    @staticmethod
    def convert_box_xywh_to_xyxy(box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]

    @staticmethod
    def segment_image(image, segmentation_mask):
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

    def segment_and_retrieve(self, image_path, search_text, threshold=0.05):
        # Load image
        image = Image.open(image_path)

        # Generate masks
        masks = self.mask_generator.generate(np.array(image))

        # Cut out all masks
        cropped_boxes = []
        for mask in masks:
            cropped_boxes.append(
                self.segment_image(image, mask["segmentation"]).crop(self.convert_box_xywh_to_xyxy(mask["bbox"]))
            )

        # Load CLIP
        model, preprocess = self.clip_model, self.clip_preprocess

        @torch.no_grad()
        def retriev(elements: list[Image.Image], search_text: str) -> int:
            preprocessed_images = [preprocess(image).to(self.clip_device) for image in elements]
            tokenized_text = clip.tokenize([search_text]).to(self.clip_device)
            stacked_images = torch.stack(preprocessed_images)
            image_features = model.encode_image(stacked_images)
            text_features = model.encode_text(tokenized_text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = 100. * image_features @ text_features.T
            return probs[:, 0].softmax(dim=0)

        # Retrieve relevant images
        scores = retriev(cropped_boxes, search_text)
        indices = [i for i, v in enumerate(scores) if v > threshold]

        # Generate overlay image
        segmentation_masks = []
        for seg_idx in indices:
            segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
            segmentation_masks.append(segmentation_mask_image)

        overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_color = (255, 0, 0, 200)

        draw = ImageDraw.Draw(overlay_image)
        for segmentation_mask_image in segmentation_masks:
            draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)
        # Overlay the masks on the original image
        result_image = Image.alpha_composite(image.convert('RGBA'), overlay_image)
        
        return result_image, segmentation_masks

    def inpaint_image(self, image_path, search_text, prompt):
        from PIL import ImageChops, ImageFilter
        from scipy import ndimage
        from diffusers import StableDiffusionInpaintPipeline
        from functools import reduce

        # Load the image
        image = Image.open(image_path)

        # Call the segment_and_retrieve method to obtain the result image with segmentation masks
        result_image, segmentation_masks = self.segment_and_retrieve(image_path, search_text)

        # Extract the segmentation masks
        # segmentation_masks = []
        # for seg_idx in range(len(result_image.getbands())):
        #     if result_image.getbands()[seg_idx] == 'A':
        #         segmentation_masks.append(result_image.getchannel(seg_idx))

        # Merge masks
        merged_mask = reduce(ImageChops.add, segmentation_masks)
        # merged_mask = ImageChops.add(segmentation_masks[0] , segmentation_masks[1])

        # Convert the binary mask image to a numpy array
        mask_array = np.array(merged_mask)

        # Define a structuring element for morphological operations
        structuring_element = ndimage.generate_binary_structure(2, 2)

        # Perform erosion to remove small isolated pixels
        eroded_mask = ndimage.binary_erosion(mask_array, structure=structuring_element, iterations=2)

        # Perform dilation to fill small holes in the object masks
        dilated_mask = ndimage.binary_dilation(eroded_mask, structure=structuring_element, iterations=2)

        # Convert the numpy array back to a PIL image
        filtered_mask = Image.fromarray(np.uint8(dilated_mask) * 255)

        image_source_for_inpaint = image.resize((512, 512))
        image_mask_for_inpaint = filtered_mask.resize((512, 512))

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )

        pipe = pipe.to("cuda")

        image_inpainting = pipe(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]

        image_inpainting = image_inpainting.resize((image.size[0], image.size[1]))

        return image_inpainting
        