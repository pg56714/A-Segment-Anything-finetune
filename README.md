# Segment-Anything-finetune

## Getting Started

```bash
conda create --name sam-finetune python=3.10 -y

conda activate sam-finetune

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Datasets Steps

1. Download from the [ViSha Dataset Source Link](https://erasernut.github.io/ViSha.html) or [SOBA_v2-Datasets](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP) and place the files inside the `datasets` folder.

2. Run the `save_json.py` script in the `tool` folder to create `sam_train.json` and `sam_test.json` and place them inside the `datasets` folder.

3. Finish.

You can use the `save_labels.py` script in the `tool` folder to verify the labels.

Organized [SOBA_v2-Datasets](https://drive.google.com/drive/folders/1561wGAf0oik7C7__3byLHBNJOIadFuMw?usp=sharing) for use with the SAM model.

### Weights

Download the weights from the following links and save them in the `weights` directory.

[ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Fine-tune

```bash
python sam_finetune.py
```

## Test

```bash
python sam_test.py
```

## Eval

```bash
python sam_eval.py
```

## Demo

```bash
python demo_app.py
```

## Source

[segment-anything](https://github.com/facebookresearch/segment-anything)

[Detect-AnyShadow](https://github.com/harrytea/Detect-AnyShadow)

[ViSha-Dataset-Source-Link](https://erasernut.github.io/ViSha.html)

[ViSha-Dataset-Link](https://drive.google.com/drive/folders/1Hp2mn_ui8I9GfxpXbLZ8zOvLlc_YJH4X)

[learn-how-to-fine-tune-the-segment-anything-model-sam](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)

[fine-tune-the-segment-anything-model-sam-colab](https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=WRQ6yd_PM_B9)

[SOBA_v2-Datasets](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP)

[SSIS](https://github.com/stevewongv/SSIS)
