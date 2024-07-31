# Segment-Anything-finetune-shadow

## Getting Started

```bash
conda create --name anyshadow python=3.10 -y

conda activate anyshadow

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Datasets Steps

1. Download from the [ViSha Dataset Source Link](https://erasernut.github.io/ViSha.html) and place the files inside the `datasets` folder.

2. Run the `save_json.py` script in the `tool` folder to create `sam_train.json` and `sam_test.json` and place them inside the `datasets` folder.

3. Finish.

You can use the `save_labels.py` script in the `tool` folder to verify the labels.

## Fine-tune

```bash
python sam_finetune.py
```

## Demo

```bash
python demo_app.py
```

## Source

[Detect-AnyShadow](https://github.com/harrytea/Detect-AnyShadow)

[ViSha-Dataset-Source-Link](https://erasernut.github.io/ViSha.html)

[ViSha-Dataset-Link](https://drive.google.com/drive/folders/1Hp2mn_ui8I9GfxpXbLZ8zOvLlc_YJH4X)

[learn-how-to-fine-tune-the-segment-anything-model-sam](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)

[fine-tune-the-segment-anything-model-sam-colab](https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=WRQ6yd_PM_B9)
