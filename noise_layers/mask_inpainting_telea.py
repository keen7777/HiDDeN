import torch
import torch.nn as nn
import numpy as np
import cv2

#class MaskInpaintingTelea(nn.Module):
class MaskInpainting(nn.Module):
    """
    use random mask to do inpaiting, give a range of the size of the mask;
    use a random number generator(seed=42) to generate different mask for each image(reproducible), 
    and use classic telea method in opencv library.(mask same as baseline method) 
    """

    def __init__(self, mask_size_range_min, mask_size_range_max, seed=42):
        super(MaskInpainting, self).__init__()
        self.mask_min = mask_size_range_min
        self.mask_max = mask_size_range_max
        self.rng = np.random.RandomState(seed)


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # don't need cover image to do repair, unlike dropout/cropout
        cover_image = noised_and_cover[1]

        #batch_size, channels, height, width
        _, _, H, W = noised_image.shape
        output_image = noised_image.clone()

        # random mask size ratio, use rng from init, decide mask size, h and w
        mask_ratio = self.rng.uniform(self.mask_min, self.mask_max)
        mask_h = int(H * mask_ratio)
        mask_w = int(W * mask_ratio)

        # position of mask
        top = self.rng.randint(1, H - mask_h - 1)
        left = self.rng.randint(1, W - mask_w - 1)

        # telea: 2d black mask, same size as images
        whole_image_mask = np.zeros((H, W), dtype=np.uint8)
        whole_image_mask[top:top+mask_h, left:left+mask_w] = 255

        # same mask for the batch, do i need to make it different for each image?

        # loop the batch
        for i in range(noised_image.shape[0]):
            # tensor -> uint8(opencv form) -> tensor

            # tensor (C,H,W) -> numpy (H,W,C)
            # non-differentiable attack, bc of detach
            img = output_image[i].detach().permute(1,2,0).cpu().numpy()

            img_uint8 = (img * 255).astype(np.uint8)

            # debug
            # print("img:", img_uint8.shape)
            # print("mask:", whole_image_mask.shape)

            result = cv2.inpaint(img_uint8, whole_image_mask, 3, cv2.INPAINT_TELEA)

            result_tensor = torch.from_numpy(result).permute(2,0,1).float() / 255.0

            output_image[i] = result_tensor.to(output_image.device)
           

        return [output_image, cover_image]
