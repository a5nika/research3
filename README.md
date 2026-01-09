# Sanskrit Character Image Generation with DDPM

This repository contains a **Denoising Diffusion Probabilistic Model (DDPM)** for generating images of Sanskrit characters. The model is trained on a **self-curated dataset** and evaluated using quantitative metrics such as **Fréchet Inception Distance (FID)**, and image samples for qualitative analysis. This work explores the impact of dataset scale on model performance through the scaling laws.

---

## Dataset

- **Source:** Self-curated from web-scraped Sanskrit text.  
- **Preprocessing:**
  1. Scraped text converted to a dataframe using TTF fonts and Grapheme library.
  2. Images of individual characters programmatically drawn.
  3. Saved into .png files 
- **Dataset Sizes Tested:**

| # Images     | % of Dataset |
|-------------:|:------------|
| 570          | 1%           |
| 13,000       | 25%         |
| 26,000       | 50%         |
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
  2. Compute FID between generated and real images, after every 5 epoch to track generation quality.
  3. Save image grids for visualization.

## Results

- **Quantitative Analysis:** Scaling the dataset significantly improves generative quality. The DDPM model was tested on four dataset sizes (570, 13,000, 26,000, 57,000 images) with proportionally reduced epochs. FID scores decreased steadily from 29.30 (1% data) to 0.0148 (100% data), indicating that larger, more diverse datasets have a greater impact on model performance than prolonged training on smaller subsets.

- **Qualitative Assessment:** At lower data tiers, generated Sanskrit graphemes exhibit fragmented strokes and broken ligatures. As dataset size increases, the model captures the fine structural details of Devanagari, producing coherent, visually distinct, and semantically legible characters. These results highlight that prioritizing high-quality, large-scale datasets is more effective than extending training on limited data for complex script synthesis.
