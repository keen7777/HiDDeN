import torch
import torch.nn as nn
import numpy as np

class MaskInpainting(nn.Module):
    """
    use random mask to do inpaiting, give a range of the size of the mask;
    use a random number generator(seed=42) to generate different mask for each image(reproducible), 
    and use the neighbor/surrounding part of the mask to fill in the blank. 
    """

    def __init__(self, mask_size_range_min, mask_size_range_max, seed=42):
        super(MaskInpainting, self).__init__()
        self.mask_min = mask_size_range_min
        self.mask_max = mask_size_range_max
        self.rng = np.random.RandomState(seed)


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        _, _, H, W = noised_image.shape
        output_image = noised_image.clone()

        # random mask size ratio, use rng from init
        mask_ratio = self.rng.uniform(self.mask_min, self.mask_max)
        mask_h = int(H * mask_ratio)
        mask_w = int(W * mask_ratio)

        # position of mask
        top = self.rng.randint(1, H - mask_h - 1)
        left = self.rng.randint(1, W - mask_w - 1)

        for b in range(noised_image.shape[0]):

            # surrounding border pixels
            neighbors = []

            # borders: 
            # top
            neighbors.append(output_image[b, :, top-1, left:left+mask_w])  
            # bottom
            neighbors.append(output_image[b, :, top+mask_h, left:left+mask_w])  
            # left
            neighbors.append(output_image[b, :, top:top+mask_h, left-1])     
            # right
            neighbors.append(output_image[b, :, top:top+mask_h, left+mask_w])       

            # flatten and concatenate
            neighbors = torch.cat([x.reshape(-1) for x in neighbors])

            # just use a mean value for baseline approach, later use opencv version?
            fill_value = neighbors.mean()

            output_image[b, :, top:top+mask_h, left:left+mask_w] = fill_value

        return [output_image, cover_image]
