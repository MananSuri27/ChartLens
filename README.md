# Chart Visual Grounding Dataset


This repository contains the benchmark used in the ACL 2025 paper "ChartLens: Fine-grained Visual Attribution in Charts". The dataset provides fine-grained visual attribution annotations for chart question-answering tasks.

## 📖 About the Paper

Our ACL 2025 paper introduces **ChartLens**, a novel approach for fine-grained visual attribution in charts that addresses hallucinations in multimodal large language models (MLLMs). The paper proposes:

- **Post-Hoc Visual Attribution**: A method to identify specific chart elements that support textual responses
- **ChartLens Algorithm**: Uses segmentation-based techniques and set-of-marks prompting for accurate attribution
- **ChartVA-Eval Benchmark**: This dataset with 1200+ samples across diverse domains


## 📊 Dataset Overview

This dataset enables research in **chart visual grounding** - the task of identifying which specific visual elements in a chart support answers to questions about that chart.

## 🗂️ Dataset Structure

The repository contains the **ChartVA-Eval** benchmark with visual attribution annotations:

```
./
├── data/
│   ├── matsa_dataset.csv           # MATSA-AITQA subset 
│   ├── chartqa_dataset.csv         # ChartQA subset 
│   └── plotqa_dataset.csv          # PlotQA subset 
├── images/
│   ├── MATSA/                      # Chart images from MATSA
│   ├── ChartQA/                    # Chart images from ChartQA
│   └── PlotQA/                     # Chart images from PlotQA
└── README.md
```

### Data Format

Each CSV contains:
- `id`: Unique sample identifier
- `question`: Question about the chart
- `answer`: Ground truth answer
- `chart_type`: Type of chart (bar/line/pie)
- `image_path`: Path to chart image
- `bboxes`: Visual attribution coordinates
  - **Line charts**: Point coordinates `[{"x1":100,"y1":50}]`
  - **Bar/Pie charts**: Bounding boxes `[{"x1":100,"y1":50,"x2":150,"y2":200}]`

## 🚀 Quick Start

### Loading the Dataset

```python
import pandas as pd
import json

# Load dataset
df = pd.read_csv('data/matsa_dataset.csv')

# Parse visual attributions
def parse_attributions(bbox_str):
    return json.loads(bbox_str) if bbox_str else []

df['attributions'] = df['bboxes'].apply(parse_attributions)

# Example usage
sample = df.iloc[0]
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
print(f"Attributions: {sample['attributions']}")
```

### Visualization Example

```python
import cv2
import matplotlib.pyplot as plt

def visualize_attributions(image_path, attributions):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for attr in attributions:
        # Draw bounding box or point
        if 'x2' in attr:  # Bounding box
            cv2.rectangle(img, (attr['x1'], attr['y1']), 
                         (attr['x2'], attr['y2']), (255, 0, 0), 2)
        else:  # Point
            cv2.circle(img, (attr['x1'], attr['y1']), 5, (255, 0, 0), -1)
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```


## 📝 Citation

If you use this dataset in your research, please cite our ACL 2025 paper:

```bibtex
@inproceedings{suri-etal-2025-chartlens,
    title = "{C}hart{L}ens: Fine-grained Visual Attribution in Charts",
    author = "Suri, Manan and Mathur, Puneet and Lipka, Nedim and 
              Dernoncourt, Franck and Rossi, Ryan A. and Manocha, Dinesh",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association 
                 for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1094/",
    pages = "22447--22462",
}
```

