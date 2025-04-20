# 🚀 Automated YOLO Dataset Generator using Grounding DINO + SAM

Generate high-quality, YOLOv8-compatible datasets with ease by leveraging the power of **Grounding DINO** for zero-shot object detection and **Segment Anything (SAM)** for precise segmentation. This tool allows you to annotate images using intuitive text prompts (e.g., "tomato", "onion", etc.), and automatically produces properly structured YOLO-format datasets complete with labeled training, validation, and testing sets, along with a `data.yaml` file for immediate use in model training.

---

## 🚀 Features

- 🔍 Zero-shot object detection using class names (e.g. "potato", "car", etc.)
- ✏️ Segmentation masks via Meta's Segment Anything
- 🌀 Automatically generates `train`, `valid`, `test` folders
- 🔢 YOLOv8-compatible `data.yaml` file
- 🚀 Easy-to-use, customizable CLI

---

## ⚙️ Installation

### 1. Clone the repository
```bash
https://github.com/yourusername/grounding-dino-yolo-dataset-generator.git
cd grounding-dino-yolo-dataset-generator
```

### 2. Install dependencies (recommended in virtualenv or conda)
```bash
pip install -r requirements.txt
```

> Make sure you have Python 3.9+ and CUDA installed.

### 3. Setup Grounding DINO

- Clone the [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) repository and download the SwinT model:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
mkdir weights
# Download the SwinT model
curl -L https://github.com/IDEA-Research/GroundingDINO/releases/download/0.1.0/groundingdino_swint_ogc.pth -o weights/groundingdino_swint_ogc.pth
```

### 4. Setup Segment Anything (SAM)

- Clone the [SAM repository](https://github.com/facebookresearch/segment-anything) and download ViT-H model:
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
# Download ViT-H SAM model
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o weights/sam_vit_h_4b8939.pth
```

> Update the paths in the script to point to your local GroundingDINO config and checkpoint, and the SAM checkpoint.

---

## 🔧 How It Works

1. Takes class names from user input (e.g., "potato, tomato")
2. Detects objects using Grounding DINO with enhanced prompts (e.g., "all tomatoes")
3. Segments each object using SAM
4. Converts boxes to YOLO format
5. Splits images and labels into `train`, `valid`, and `test` folders
6. Creates a `data.yaml` file for training in YOLOv8

---

## 🚧 Usage

### 1. Place your images inside a folder (e.g. `Potato/`)

### 2. Run the script
```bash
python annotate.py
```

You'll be prompted to:
- Enter class names (comma separated)
- Enter name for the output dataset folder

The script will:
- Process all images
- Detect, segment and convert to YOLO format
- Create folder structure:  
```
dataset_name/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## 🌐 Example `data.yaml`
```yaml
train: /path/to/train/images
val: /path/to/valid/images

nc: 3
names: ['tomato', 'potato', 'onion']
```

---

## 📄 License
MIT License

---

## 👁️ Credits
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---
