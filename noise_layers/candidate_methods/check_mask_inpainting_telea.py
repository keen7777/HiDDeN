import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from pathlib import Path


def telea_inpainting(image, mask_min=0.01, mask_max=0.05, seed=13):
    """
    image: torch.Tensor (C, H, W), range [0,1]

    return:
        result_tensor: torch.Tensor (C, H, W), range [0,1]
        mask: np.ndarray (H, W), uint8, values 0 or 255
    """

    rng = np.random.RandomState(seed)

    C, H, W = image.shape
    output = image.clone()

    # ===== 1. Generate random rectangular mask =====
    mask_ratio = rng.uniform(mask_min, mask_max)
    mask_h = int(H * mask_ratio)
    mask_w = int(W * mask_ratio)

    top = rng.randint(1, H - mask_h - 1)
    left = rng.randint(1, W - mask_w - 1)

    # OpenCV needs a mask of shape (H, W)
    # 0   = keep original region
    # 255 = region to be inpainted
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[top:top + mask_h, left:left + mask_w] = 255

    # ===== 2. Tensor -> NumPy =====
    img = output.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img_uint8 = (img * 255).astype(np.uint8)

    # ===== 3. OpenCV Telea inpainting =====
    result = cv2.inpaint(
        img_uint8,
        mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    # ===== 4. NumPy -> Tensor =====
    result_tensor = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0

    print(f"mask: top={top}, left={left}, h={mask_h}, w={mask_w}")

    return result_tensor, mask


if __name__ == "__main__":
    input_path = "../../images/train/train_class/000000000009.jpg"

    output_dir = Path("validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "inpainting_telea_000000000009_small.png"
    mask_path = output_dir / "mask_000000000009_small.png"

    # ===== Read image =====
    img = Image.open(input_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img)

    # ===== Run inpainting =====
    result_tensor, mask = telea_inpainting(img_tensor)

    # ===== Save inpainted result =====
    to_pil = T.ToPILImage()
    result_img = to_pil(result_tensor)
    result_img.save(output_path)

    # ===== Save mask image =====
    # White area = masked / inpainted region
    # Black area = unchanged region
    mask_img = Image.fromarray(mask)
    mask_img.save(mask_path)

    print("Done!")
    print("Saved inpainted image to:", output_path)
    print("Saved mask image to:", mask_path)