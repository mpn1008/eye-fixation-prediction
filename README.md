# CV2 Architecture for Eye-fixation Prediction Model
 
## Task Definition
 
Given a scene image (RGB), predict a fixation density map (grayscale) where pixel intensity represents how likely humans are to look at that location.
 
This is a **dense prediction** task: every output pixel needs a value, at the same spatial resolution as the input.
 
---

## My first thoughts of the task:

Given the training data, it is quite clear that I have to implement a pre-trained net because there are only about 3000 data points.

I tried both ResNet50 and ResNet101 as the encoder. ResNet101 gives slightly better validation result (ResNet50: 0.8146, ResNet101: 0.8192) but the convergence time is higher for ResNet101 (around epoch 25), ResNet50 converges around epoch 15.



## Why CNN (not a Transformer)?
 
Eye fixation prediction benefits from two properties that CNNs provide by design:
 
- **Translation equivariance** — a salient region (a face, text, bright object) is salient regardless of where it appears in the image.
- **Local-to-global hierarchy** — edges → textures → objects → scene context, which mirrors how the human visual system computes salience bottom-up.
 
Transformers (e.g. ViT, TranSalNet) need large datasets to learn these priors from scratch. With ~3 000 training samples, a pretrained CNN backbone is a significantly safer and more data-efficient choice.
 
---
 
## Why Encoder-Decoder (U-Net Style)?
 
Fixation maps require two contradictory things simultaneously:
 
| Requirement | Source |
|---|---|
| **High-level semantics** — *what* is salient (faces, text, motion, contrast) | Deep encoder layers with large receptive field |
| **Spatial precision** — *where exactly* the fixation peak is | Fine-grained early feature maps |
 
A plain encoder progressively shrinks spatial resolution (224 → 7). Reconstructing a sharp map from that bottleneck alone produces blurry outputs.
 
**Skip connections** (U-Net pattern) solve this by concatenating encoder feature maps at each resolution stage back into the decoder. The decoder recovers spatial detail without having to re-learn it from the bottleneck.
 
---
 
## Why ResNet-50 as the Backbone?
 
| Reason | Detail |
|---|---|
| **ImageNet pretraining** | The encoder already recognises faces, objects, text — the exact things humans fixate on |
| **Four residual stages** | Maps cleanly to four decoder blocks with skip connections |
| **Stability** | Well-studied, no training instability, easy to fine-tune |
 
---
 
## Why the Combined Loss (KL + CC + MSE)?
 
No single loss function adequately captures what "a good saliency map" means:
 
| Loss | Penalises |
|---|---|
| **KL divergence** | Wrong probability mass distribution — predicted density in the wrong regions |
| **1 − CC** (Pearson correlation) | Low correlation even when the shape is plausible — forces the model to match relative intensities across the map |
| **MSE** | Large absolute pixel errors — keeps the predicted values numerically close to ground truth |
 
This combination is the de-facto standard used in published saliency benchmarks (MIT300, SALICON, CAT2000).
 
---
 
## Evaluation Metrics
 
| Metric | Interpretation |
|---|---|
| **CC** (Pearson Correlation Coefficient) | How well predicted and ground-truth maps correlate linearly. Primary checkpoint criterion. Higher is better. |
| **KL Divergence** | How different the predicted density distribution is from ground truth. Lower is better. |
| **NSS** (Normalised Scanpath Saliency) | Average normalised saliency value at actual fixation locations. Higher is better. |
 
---
 
## Alternatives Considered
 
| Alternative | Reason Not Used |
|---|---|
| Plain encoder + bilinear upsample | Loses spatial detail; produces blurry maps |
| Encoder-only with FC head | Ignores spatial structure entirely |
| Transformer (ViT / TranSalNet) | Needs significantly more data; marginal gain at this dataset scale |
| Separate branches for low/high freq | Added complexity with no clear benefit over skip connections |
 
---
 
## Model Summary
 
```
Input  (B, 3, 224, 224)
           │
    ResNetEncoder  (pretrained ResNet-50)
           ├── s1   64ch  112×112   stem
           ├── s2  256ch   56×56    layer1
           ├── s3  512ch   28×28    layer2
           ├── s4 1024ch   14×14    layer3
           └── s5 2048ch    7×7     layer4
                   │
    SaliencyDecoder  (U-Net skip connections)
           bottleneck  → 256ch   7×7
           block3 + s4 → 128ch  14×14
           block2 + s3 →  64ch  28×28
           block1 + s2 →  32ch  56×56
           block0 + s1 →  16ch 112×112
           upsample    →       224×224
           head (1×1 conv + sigmoid)
                   │
Output (B, 1, 224, 224)  ∈ [0, 1]
```
 