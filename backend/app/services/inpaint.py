import cv2
import numpy as np

class InpaintService:
    def heal_text_areas(self, img_cv2, text_metadata):
        """
        Removes original text while preserving the background texture/color.
        """
        mask = np.zeros(img_cv2.shape[:2], dtype=np.uint8)
        
        for block in text_metadata:
            coords = block.get('coords', {})
            x, y, w, h = int(coords.get('x', 0)), int(coords.get('y', 0)), int(coords.get('w', 0)), int(coords.get('h', 0))
            
            # Expanded mask for better blending at edges
            padding = 3
            cv2.rectangle(mask, (x-padding, y-padding), (x+w+padding, y+h+padding), 255, -1)
        
        # Telea works well for restoring document textures and colored blocks
        inpainted_img = cv2.inpaint(img_cv2, mask, 3, cv2.INPAINT_TELEA)
        
        # Check for colored blocks. If the inpainted area is too different 
        # from the background, we blend it more aggressively.
        # (This handles text in blue bars, red boxes, etc.)
        for block in text_metadata:
            coords = block.get('coords', {})
            x, y, w, h = int(coords.get('x', 0)), int(coords.get('y', 0)), int(coords.get('w', 0)), int(coords.get('h', 0))
            
            # sample the edge pixels to determine the background color
            edge_padding = 5
            edge_area = inpainted_img[max(0, y-edge_padding):min(img_cv2.shape[0], y+h+edge_padding),
                                     max(0, x-edge_padding):min(img_cv2.shape[1], x+w+edge_padding)]
            if edge_area.size > 0:
                # get median color
                median_bgr = np.median(edge_area, axis=(0, 1))
                # blend original inpainted with median color to enforce solid colored blocks
                cv2.rectangle(inpainted_img, (x, y), (x+w, y+h), median_bgr, -1)

        return inpainted_img, mask

inpaint_engine = InpaintService()