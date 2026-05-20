### this is a file contain all the commands and comparison of different models.

# test_model:
# command:
python test_model.py -c "runs/3k_baseline 2026.05.13--12-17-40/checkpoints/3k_baseline--epoch-400.pyt" -s images/train/train_class/000000000009.jpg -o "runs/3k_baseline 2026.05.13--12-17-40/options-and-config.pickle"
# baseline: 
test_model.py:47: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(args.checkpoint_file)
tensor(33.8354, device='cuda:0')
original: [[1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0.
  0. 0. 0. 1. 0. 1.]]
decoded : [[1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0.
  0. 0. 0. 1. 0. 1.]]
error : 0.000

# validate-trained-models:

python validate-trained-models.py \
  -d images \
  -r runs

python validate-trained-models.py \
  -d images \
  -r runs \
  --run-name "3k_baseline 2026.05.13--12-17-40"

python validate-trained-models.py \
  -d images \
  -r runs \
  --run-name "3k_inpainting_telea 2026.05.17--14-15-56"

python validate-trained-models.py \
  -d images \
  -r runs \
  --run-name "3k_inpainting 2026.05.17--20-40-50"

python validate-trained-models.py \
  -d images \
  -r runs \
  --run-name "3k_jpeg 2026.05.15--15-48-12"


## sweep: baseline model to different attack:
# base -> dropout
python sweep.py \
  -d images \
  -r runs \
  --run-name "3k_baseline 2026.05.13--12-17-40" \
  --attack dropout \
  --min 0.1 --max 0.9 --steps 9

# base-> telea inpainting 
python sweep.py \
  -d images \
  -r runs \
  --run-name "3k_baseline 2026.05.13--12-17-40" \
  --attack teleamaskinpainting \
  --min 0.1 --max 0.9 --steps 9

# different strengh:
# base-> telea inpainting 
python sweep.py \
  -d images \
  -r runs \
  --run-name "3k_baseline 2026.05.13--12-17-40" \
  --attack teleamaskinpainting \
  --min 0.01 --max 0.09 --steps 9

# base -> resize
python sweep.py   -d images   -r runs   --run-name "3k_baseline 2026.05.13--12-17-40"   --attack resize   --min 0.1 --max 0.9 --steps 9

### self -> self
# resize:
python sweep.py   -d images   -r runs   --run-name "3k_resize 2026.05.14--17-34-33"   --attack resize   --min 0.1 --max 0.9 --steps 9