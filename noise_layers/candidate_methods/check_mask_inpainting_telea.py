import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T


def telea_inpainting(image, mask_min=0.2, mask_max=0.4, seed=42):
    """
    image: torch.Tensor (C, H, W), range [0,1]
    return: torch.Tensor (C, H, W)
    """

    rng = np.random.RandomState(seed)

    C, H, W = image.shape
    output = image.clone()

    # ===== 1. 生成随机mask =====
    mask_ratio = rng.uniform(mask_min, mask_max)
    mask_h = int(H * mask_ratio)
    mask_w = int(W * mask_ratio)

    top = rng.randint(1, H - mask_h - 1)
    left = rng.randint(1, W - mask_w - 1)

    # OpenCV需要(H,W)的mask
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[top:top+mask_h, left:left+mask_w] = 255

    # ===== 2. tensor -> numpy =====
    img = output.permute(1, 2, 0).cpu().numpy()   # (H,W,C)
    img_uint8 = (img * 255).astype(np.uint8)

    # ===== 3. OpenCV inpainting =====
    result = cv2.inpaint(
        img_uint8,
        mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    # ===== 4. numpy -> tensor =====
    result_tensor = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0

    print(f"mask: top={top}, left={left}, h={mask_h}, w={mask_w}")

    return result_tensor


if __name__ == "__main__":
    input_path = "../../images/train/train_class/000000000009.jpg"   #first image
    output_path = "validation_results/inpainting_telea_000000000009.png"

    # ===== 读取图片 =====
    img = Image.open(input_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img)

    # ===== 执行 inpainting =====
    result_tensor = telea_inpainting(img_tensor)

    # ===== 保存 =====
    to_pil = T.ToPILImage()
    result_img = to_pil(result_tensor)
    result_img.save(output_path)

    print("Done! Saved to", output_path)