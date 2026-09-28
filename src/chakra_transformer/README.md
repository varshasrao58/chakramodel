# ChakraTransformer (Research Track)

Welcome to the **Research Track** of ChakraModel.

While the `src/` directory contains the CNN-based YOLO + PraNet pipeline optimized for real-time 49+ FPS inference, this folder contains the architecture designed for maximum accuracy on benchmark leaderboards.

## Why a Vision Transformer?
Vision Transformers (ViTs) utilize self-attention mechanisms to capture global context across an image far better than standard Convolutional Neural Networks (CNNs). This allows the model to distinguish extremely subtle features, such as completely flat polyps (Paris Classification Type 0-IIb) that blend deeply into the mucosal background.

## Decoder design

The segmenter now follows the useful dense-prediction pattern from
[EndoViT](https://github.com/DominikBatic/EndoViT): it captures four transformer
depths, projects them into a multi-scale pyramid, and fuses them with residual
refinement blocks before producing the mask. This preserves ChakraModel's
`timm` ImageNet backbone and binary Kvasir-SEG training contract; EndoViT's
MAE ViT-B/16 checkpoint is not loaded automatically because it is a different
backbone and checkpoint format.

## Trade-offs
- **Pros:** Highest possible Dice score and precision. Excellent for post-procedure auditing where live latency is not an issue.
- **Cons:** Significantly slower inference (often <15 FPS on laptop hardware) and high VRAM consumption.

## Usage
1. Ensure you have installed the transformer dependencies:
   ```bash
   pip install timm
   ```
2. You can inspect the model architecture:
   ```bash
   python transformer_segmenter.py
   ```
3. To train the transformer:
   ```bash
   # T4/16 GB-friendly configuration; effective batch size is 16.
   python train_transformer.py --epochs 50 --batch-size 2 --grad-accum 8 --workers 4
   ```

   The trainer enables CUDA AMP by default, saves the best weights to
   `weights/chakra_transformer_best.pth`, and writes resumable state to
   `weights/chakra_transformer_latest.pth`. Resume with
   `--resume weights/chakra_transformer_latest.pth`.

## GitHub Actions GPU training

Run `.github/workflows/train-transformer-gpu.yml` with **Actions → Train
ChakraTransformer on GPU → Run workflow**. Select the GPU runner label enabled
for your repository or organization. GitHub larger-runner labels are
account-specific, so replace the default `ubuntu-24.04-gpu` input if your
runner uses a custom label. The workflow downloads Kvasir-SEG, verifies
`nvidia-smi`, trains with AMP, and uploads both best and resumable checkpoints
as workflow artifacts.
