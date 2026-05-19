from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
import torch

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1.0 / mse)

def compute_ssim(img1, img2):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    score = ssim(img1, img2)
    return score