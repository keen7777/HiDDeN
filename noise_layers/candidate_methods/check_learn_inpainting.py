import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.insert(0, PROJECT_ROOT)

from noise_layers.learnable_inpainting import LearnableInpainting


def learnable_inpainting_check(
    image,
    mask_min=0.2,
    mask_max=0.4,
    seed=42,
    soft_mask=False
):
    """
    中文：
        单独测试 LearnableInpainting 的前向结果。
        输入一张图片，生成 multi-rectangle union mask，
        然后用 CorruptionNet 填充 mask 区域。

    English:
        Standalone test for LearnableInpainting.
        It generates a multi-rectangle union mask and
        applies the learnable corruption network.

    image:
        torch.Tensor, shape (C, H, W), range [0,1]

    return:
        corrupted_tensor: torch.Tensor, shape (C,H,W)
        mask_tensor:      torch.Tensor, shape (1,H,W)
        masked_tensor:    torch.Tensor, shape (C,H,W)
    """

    # --------------------------------------------------
    # Add batch dimension
    # 中文：LearnableInpainting 需要 batch 输入 [B,C,H,W]
    # English: LearnableInpainting expects batched input [B,C,H,W]
    # --------------------------------------------------
    image_batch = image.unsqueeze(0)

    # --------------------------------------------------
    # cover_image 在这个测试里不重要
    # 为了保持 HiDDeN 接口，直接用 image 自己作为 cover
    #
    # cover_image is not important in this standalone test.
    # We use the same image as cover to keep the HiDDeN-style interface.
    # --------------------------------------------------
    cover_batch = image_batch.clone()

    # --------------------------------------------------
    # Create learnable inpainting layer
    # 中文：这里会使用你 learnable_inpainting.py 里的 generate_mask()
    # English: This uses generate_mask() inside your LearnableInpainting class.
    # --------------------------------------------------
    layer = LearnableInpainting(
        min_ratio=mask_min,
        max_ratio=mask_max,
        hidden_channels=32,
        soft_mask=soft_mask
    )

    # --------------------------------------------------
    # Fix random seed for reproducibility
    # 中文：保证每次生成的 mask 尽量一致，方便 debug
    # English: Make the random mask reproducible for debugging.
    # --------------------------------------------------
    layer.rng = np.random.RandomState(seed)

    layer.eval()

    with torch.no_grad():

        # --------------------------------------------------
        # Manually generate mask for visualization
        # 中文：单独生成 mask，方便保存和检查
        # English: Generate mask separately for visualization.
        # --------------------------------------------------
        B, C, H, W = image_batch.shape

        mask = layer.generate_mask(
            B=B,
            H=H,
            W=W,
            device=image_batch.device
        )

        if soft_mask:
            mask = layer.soften_mask(mask)

        # --------------------------------------------------
        # Apply corruption network directly
        # 中文：这里直接调用 corrnet，方便确认 CorruptionNet 是否正常
        # English: Directly call corrnet for clearer debugging.
        # --------------------------------------------------
        corrupted = layer.corrnet(
            image_batch,
            mask
        )

        # --------------------------------------------------
        # Also create masked image
        # 中文：显示“挖洞后”的图像
        # English: Image after removing masked region.
        # --------------------------------------------------
        masked = image_batch * (1 - mask)

    # --------------------------------------------------
    # Print statistics
    # 中文：检查输出范围，防止出现 [-1,1] 或异常颜色
    # English: Check output range to avoid abnormal values.
    # --------------------------------------------------
    print("Input:")
    print("  min:", image_batch.min().item())
    print("  max:", image_batch.max().item())
    print("  mean:", image_batch.mean().item())

    print("Mask:")
    print("  min:", mask.min().item())
    print("  max:", mask.max().item())
    print("  mean / coverage:", mask.mean().item())

    print("Corrupted:")
    print("  min:", corrupted.min().item())
    print("  max:", corrupted.max().item())
    print("  mean:", corrupted.mean().item())

    return (
        corrupted.squeeze(0),
        mask.squeeze(0),
        masked.squeeze(0)
    )


if __name__ == "__main__":

    input_path = "images/train/train_class/000000000009.jpg"

    os.makedirs("validation_results", exist_ok=True)

    output_corrupted_path = "noise_layers/candidate_methods/validation_results/learnable_inpainting_corrupted_000000000009.png"
    output_mask_path = "noise_layers/candidate_methods/validation_results/learnable_inpainting_mask_000000000009.png"
    output_masked_path = "noise_layers/candidate_methods/validation_results/learnable_inpainting_masked_000000000009.png"

    # --------------------------------------------------
    # Read image
    # 中文：读取输入图片并转为 [0,1] tensor
    # English: Load image and convert it to [0,1] tensor.
    # --------------------------------------------------
    img = Image.open(input_path).convert("RGB")

    transform = T.ToTensor()
    img_tensor = transform(img)

    # --------------------------------------------------
    # Run learnable inpainting check
    # --------------------------------------------------
    result_tensor, mask_tensor, masked_tensor = learnable_inpainting_check(
        img_tensor,
        mask_min=0.2,
        mask_max=0.4,
        seed=42,
        soft_mask=False
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    to_pil = T.ToPILImage()

    result_img = to_pil(torch.clamp(result_tensor, 0, 1))
    result_img.save(output_corrupted_path)

    # mask: [1,H,W] -> PIL grayscale
    mask_img = to_pil(mask_tensor)
    mask_img.save(output_mask_path)

    masked_img = to_pil(torch.clamp(masked_tensor, 0, 1))
    masked_img.save(output_masked_path)

    print("Done!")
    print("Saved corrupted image to:", output_corrupted_path)
    print("Saved mask image to:", output_mask_path)
    print("Saved masked image to:", output_masked_path)