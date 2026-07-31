import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def inpainting(image, mask_min=0.01, mask_max=0.05, seed=13):
    """
    image: torch.Tensor (C, H, W)
    return: torch.Tensor (C, H, W)
    """

    rng = np.random.RandomState(seed)

    C, H, W = image.shape
    output = image.clone()

    # random mask size
    mask_ratio = rng.uniform(mask_min, mask_max)
    mask_h = int(H * mask_ratio)
    mask_w = int(W * mask_ratio)

    # random mask position
    top = rng.randint(1, H - mask_h - 1)
    left = rng.randint(1, W - mask_w - 1)

    # get all surrounding neighbor pixels
    neighbors = []

    neighbors.append(output[:, top-1, left:left+mask_w])        # top
    neighbors.append(output[:, top+mask_h, left:left+mask_w])   # bottom
    neighbors.append(output[:, top:top+mask_h, left-1])         # left
    neighbors.append(output[:, top:top+mask_h, left+mask_w])    # right

    neighbors = torch.cat([x.reshape(-1) for x in neighbors])

    # mean value(almost gray)
    fill_value = neighbors.mean()

    output[:, top:top+mask_h, left:left+mask_w] = fill_value

    print(f"filled value={fill_value}")

    print(f"mask: top={top}, left={left}, h={mask_h}, w={mask_w}");

    return output


if __name__ == "__main__":
    # read img
    input_path = "../../images/train/train_class/000000000009.jpg"   #first image
    output_path = "validation_results/inpainting_000000000009_small.png"

    img = Image.open(input_path).convert("RGB")

    transform = T.ToTensor()
    img_tensor = transform(img)   # (C, H, W)

    # apply the method, and store them to result
    result_tensor = inpainting(img_tensor)
  

    # save result img
    to_pil = T.ToPILImage()
    result_img = to_pil(result_tensor)
    result_img.save(output_path)
    print("Done! Saved to", output_path)