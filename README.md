# Sanskrit Character Image Generation with DDPM

This repository contains a **Denoising Diffusion Probabilistic Model (DDPM)** for generating images of Sanskrit characters. The model is trained on a **self-curated dataset** and evaluated using **Fréchet Inception Distance (FID)**.

---

## Dataset

- **Source:** Self-curated from web-scraped Sanskrit text.  
- **Preprocessing:**
  1. Scraped text converted to a dataframe using TTF fonts and Grapheme library.
  2. Images of individual characters programmatically drawn.  
- **Dataset Sizes Tested:**

| # Images     | % of Dataset |
|-------------:|:------------|
| 570          | 1%           |
| 13,000       | ~23%         |
| 26,000       | ~45%         |
| 57,000       | 100%         |

---

## Model Architecture

- **Base Model:** 2D UNet (encoder-decoder)  
- **Details:**
  - `in_channels=1`, `out_channels=1`
  - 4 downsampling and 4 upsampling blocks
  - Attention blocks applied in selected layers (`AttnDownBlock2D`, `AttnUpBlock2D`)  
- **Scheduler:** DDPM with 1000 timesteps  

---

## Training

- **Loss:** Mean Squared Error (MSE)  
- **Optimizer:** AdamW  
- **Learning Rate Scheduler:** Cosine schedule with warmup  
- **Epochs & Batch Sizes:**

| Dataset Size | Epochs | Batch Size |
|-------------:|:------:|:----------:|
| 570 images   | 150    | 64         |
| 13,000 imgs  | 80     | 64         |
| 26,000 imgs  | 40     | 64         |
| 57,000 imgs  | 20     | 64         |

- **Data Augmentation:** Resize, Grayscale, Normalize (-1 to 1)  
- **Utilities:** Gradient accumulation, Mixed precision (`fp16`), Checkpoints, Sample images saved every `n` epochs  

---

## Evaluation

- **Metric:** Fréchet Inception Distance (FID)  
- **Procedure:**
  1. Generate samples using the trained model.
  2. Compute FID between generated and real images.
  3. Save image grids for visualization.

**Example FID evaluation code:**

```python
score = evaluate_with_fid(config, epoch, pipeline, train_dataloader, device)
print(f"Epoch {epoch} | FID Score: {score:.4f}")
