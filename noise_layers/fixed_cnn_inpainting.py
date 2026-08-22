
import torch
import torch.nn as nn
import numpy as np

# fixed CNN Corruption Network
class CorruptionNet(nn.Module):

    def __init__(self, hidden_channels=32):

        super().__init__()

        self.net = nn.Sequential(

            
            # Input channels:
            #   masked RGB image : 3 channels
            #   binary mask      : 1 channel
            #
            # Total = 4 channels
            nn.Conv2d(4, hidden_channels, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),

            nn.ReLU(inplace=True),

            
            # Output:
            # predicted RGB filled region
            
            nn.Conv2d(hidden_channels, 3, 3, padding=1),

            
            # Sigmoid is usually safer for [0,1] images
            # than Tanh().
            
            nn.Sigmoid()
        )

    def forward(self, image, mask):

        
        # Remove masked region from image
        #
        # image shape:
        #   [B, 3, H, W]
        #
        # mask shape:
        #   [B, 1, H, W]
        #
        # masked result:
        #   masked area -> 0
        
        masked = image * (1 - mask)

        
        # Concatenate image + mask
        #
        # Result shape:
        #   [B, 4, H, W]
        
        inp = torch.cat([masked, mask], dim=1)

        
        # Predict filled content
        
        pred = self.net(inp)

        
        # Blend prediction into masked area
        #
        # unmasked area:
        #   keep original image
        #
        # masked area:
        #   replace with predicted content
        
        out = image * (1 - mask) + pred * mask

        return out

# Learnable Inpainting Layer
class FixedCNNInpainting(nn.Module):

    """
    Differentiable fixed inpainting corruption layer.
    Main idea:

    Instead of using non-differentiable classical inpainting methods such as:
        - Telea
        - PatchMatch
        - OpenCV inpaint

    we train a small CNN to imitate an inpainting attack.

    This keeps the corruption differentiable, so gradients
    can flow through the noise layer during training.

    Input
    --------------------------------------------------------
    forward([
        noised_image,
        cover_image
    ])
    Output
    --------------------------------------------------------
    [
        corrupted_image,
        cover_image
    ]
    Compatible with HiDDeN pipeline.
    """

    def __init__(
        self,
        min_ratio,
        max_ratio,
        hidden_channels=32,
        soft_mask=True,

        
        # multi-mask settings
        
        min_mask_size=8,
        max_aspect_ratio=4.0,
        randomize_ratio=True
    ):

        super().__init__()

        
        # target mask coverage range
        # e.g. 0.1 ~ 0.5
        
        self.mask_min = min_ratio
        self.mask_max = max_ratio

        self.soft_mask = soft_mask

        
        # rectangle mask settings
        
        self.min_mask_size = min_mask_size
        self.max_aspect_ratio = max_aspect_ratio
        self.randomize_ratio = randomize_ratio

        # reproducible RNG
        self.rng = np.random.RandomState(42)

        # learnable corruption CNN
        self.corrnet = CorruptionNet(
            hidden_channels=hidden_channels
        )

    # =====================================================
    # Generate multi-rectangle union masks
    # =====================================================
    def generate_mask(self, B, H, W, device):

        """
        Generate differentiable-style random masks.

        这个函数生成：多个随机矩形 mask然后把它们合并成 一个 union mask
        最终输出：[B, 1, H, W]的 tensor。
        这种 mask 分布更接近真实 inpainting 场景，比随机像素 mask 更合理。

        This function generates:multiple random rectangle masks and merges them into: one union mask producing a tensor of shape:
        [B, 1, H, W]
        This mask distribution is much closer to realistic inpainting attacks.
        """

        batch_masks = []

        for _ in range(B):

            
            # Decide target coverage ratio
            # e.g:  0.15 means 15% masked pixels
            
            if self.randomize_ratio:

                mask_ratio = self.rng.uniform(
                    self.mask_min,
                    self.mask_max
                )

            else:
                mask_ratio = self.mask_max

            target_coverage = mask_ratio    
            # Union mask:
            # 0 = keep
            # 1 = masked
            mask_union = np.zeros(
                (H, W),
                dtype=np.float32
            )

            max_w = W - 4
            max_h = H - 4

            iters = 0
            max_iters = 300

            # Keep adding rectangles until enough area is covered
            while (
                mask_union.mean() < target_coverage
                and iters < max_iters
            ):

                iters += 1
                
                # Random aspect ratio
                # e.g: wide rectangle; tall rectangle;square
                
                aspect_ratio = self.rng.uniform(
                    1 / self.max_aspect_ratio,
                    self.max_aspect_ratio
                )

                
                # Random rectangle area               
                area = self.rng.uniform(
                    self.min_mask_size ** 2,
                    H * W * 0.1
                )
                
                # Convert area + aspect ratio
                # into width/height
                
                w = int(np.sqrt(area * aspect_ratio))
                h = int(np.sqrt(area / aspect_ratio))

                w = max(1, min(w, max_w))
                h = max(1, min(h, max_h))

                if max_w - w <= 1 or max_h - h <= 1:
                    continue
                
                # Random rectangle position
                
                left = self.rng.randint(1, max_w - w)
                top = self.rng.randint(1, max_h - h)

                
                # Apply rectangle into union mask               
                mask_union[
                    top:top+h,
                    left:left+w
                ] = 1.0

            batch_masks.append(mask_union)

        # Convert:
        # [B,H,W]
        # ->
        # [B,1,H,W]
        
        mask = np.stack(batch_masks)

        mask_tensor = torch.tensor(
            mask,
            dtype=torch.float32,
            device=device
        )

        mask_tensor = mask_tensor.unsqueeze(1)

        return mask_tensor

    # Optional soft mask
    def soften_mask(self, mask):

        """
        Optional soft mask smoothing.
        给 mask 加一点随机浮动，避免边界过于硬。有时能让训练更稳定。
        Add small random noise to mask values to soften hard edges.
        This can sometimes improve training stability.
        """

        if not self.soft_mask:
            return mask

        noise = torch.rand_like(mask) * 0.1

        return torch.clamp(mask + noise, 0, 1)

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, noised_and_cover):

        
        # unpack input    
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        device = noised_image.device

        self.corrnet = self.corrnet.to(device)

        B, C, H, W = noised_image.shape

        
        # Generate random union masks
        mask = self.generate_mask(
            B,
            H,
            W,
            device
        )

        
        # Optional softening    
        mask = self.soften_mask(mask)

        
        # Learnable differentiable corruption
        
        corrupted = self.corrnet(
            noised_image,
            mask
        )   
        # Return HiDDeN-compatible format     
        return [corrupted, cover_image]