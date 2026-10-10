<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=LaneNet&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Learned%20Label%20Assignment%20for%20Curved%20Lane%20Detection&descAlignY=60&descSize=18" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1200&color=00E5FF&center=true&vCenter=true&width=720&lines=%F0%9F%9B%A3%EF%B8%8F+Replacing+handcrafted+cost+functions+with+a+neural+network;%F0%9F%93%88+Up+to+%2B2.96%25+F1+on+curved+lanes+(CULane);%F0%9F%A7%A0+Plug-in+module+for+CLRNet%3A+ResNet34+%7C+ResNet101+%7C+DLA34" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-0.4.6-orange?style=for-the-badge)
![MMCV](https://img.shields.io/badge/MMCV-1.2.5-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*Computer Vision project under **Dr. Himanshu Kumar**, IIT Jodhpur · Feb – May 2025*

</div>

---

## ✨ Overview

Anchor-based lane detectors like [CLRNet](https://github.com/Turoad/CLRNet) decide which predicted lane matches which ground-truth lane using a **handcrafted cost function**. That works on typical roads, but it struggles on **curved lanes, occlusions and tricky lighting**.

**LaneNet replaces that handcrafted step with a small neural network that *learns* to match predictions to ground truth.** It plugs into CLRNet without changing the core architecture, and it is pretrained on a curve-focused subset of CULane so it specializes in difficult geometry, while staying on par with the baseline on standard scenes.

| 🎯 Result | |
|---|---|
| ResNet34 backbone | **+2.8%** F1 on curved lanes |
| ResNet101 backbone | **+2.3%** F1 on curved lanes |
| DLA34 backbone | **+2.96%** F1 on curved lanes |
| Bonus | Higher confidence scores, allowing stricter filtering for better precision |

---

## 🎬 Demo

<p align="center">
  <img width="759" alt="LaneNet demo strip" src="https://github.com/user-attachments/assets/fea3c2ec-ba93-4893-8f34-c57b5891c12e" />
</p>

<p align="center">
  <img width="653" alt="LaneNet results" src="https://github.com/user-attachments/assets/cbed602f-ea03-49c6-9450-f4d7ab9375bc" />
</p>

---

## 💡 Why LaneNet?

Handcrafted label assignment:
- 📉 works in a **low-dimensional** cost space
- 🧱 is **inflexible** in unseen scenarios
- 🌀 falls short on **curves, occlusion and lighting changes**

LaneNet's answer is a **fully connected network that predicts a match probability** for every (predicted lane, ground-truth lane) pair, which gives the detector a richer view of road geometry.

---

## 🧩 Key Features

- 🧠 **Learned label assignment:** a fully connected network scores each prediction/ground-truth pair
- 🔌 **Drop-in CLRNet integration:** no change to CLRNet's core design
- 🌀 **Curve-focused pretraining:** trained on **8,677 curved-lane images** drawn from CULane
- 🏗️ **Multiple backbones:** ResNet34, ResNet101 and DLA34
- 🎯 **Higher-confidence predictions** for stricter, cleaner filtering

---

## 🏗️ Architecture

<p align="center">
  <img width="904" alt="LaneNet architecture" src="https://github.com/user-attachments/assets/75a2c4af-1638-4c3b-9f27-1744580e0e3a" />
</p>

| Stage | What happens |
|---|---|
| 1️⃣ **Backbone + FPN** | A ResNet or DLA backbone with an FPN extracts multi-scale lane features |
| 2️⃣ **Lane priors & ROIGather** | Anchor-based lane priors guide pooling of lane features |
| 3️⃣ **Prediction head** | Outputs classification scores, geometric parameters and auxiliary segmentation maps |
| 4️⃣ **LaneNet matching module** | Predicts **match probabilities** per prediction/ground-truth pair, replacing the handcrafted cost function |

---

## 📊 Results (Curve F1 on CULane)

| Backbone | Model | Curve F1 | Δ |
|---|---|---|---|
| ResNet34 | CLRNet | 72.77 | – |
| | **CLRNet + LaneNet** | **75.57** | **+2.8%** |
| ResNet101 | CLRNet | 75.57 | – |
| | **CLRNet + LaneNet** | **77.87** | **+2.3%** |
| DLA34 | CLRNet | 74.13 | – |
| | **CLRNet + LaneNet** | **77.09** | **+2.96%** |

📄 Full methodology, experiments and analysis: **[Project Report (PDF)](LaneNET_final.pdf)**

<details>
<summary><b>What the report covers (click to expand)</b></summary>
<br/>

- Background and motivation for LaneNet
- How it integrates into CLRNet
- Curve-focused dataset preparation
- Training procedure and hyperparameter choices
- Quantitative and qualitative analysis
- Comparison with state-of-the-art methods
- Future work

</details>

---

## 🗂️ Dataset

Built on the **[CULane](https://xingangpan.github.io/projects/CULane.html)** benchmark:

| Subset | Description | Images |
|---|---|---|
| Training | Standard lane images | ~88K |
| Validation | Standard lane images | ~9.7K |
| Test | 9 scenarios (normal, crowd, dazzle, shadow, curve, …) | ~34K |
| 🌀 **Curve subset** | Curved-lane images used to pretrain LaneNet | **8,677** |

Download CULane, extract it to `$CULANEROOT`, and link it into the project:

```bash
cd LaneNet-Curved-Lane-Detection
mkdir -p data
ln -s $CULANEROOT data/CULane
```

Expected layout:

```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Vishrut108/LaneNet-Curved-Lane-Detection.git
cd LaneNet-Curved-Lane-Detection

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage

**Train**
```bash
python main.py configs/clrnet/clr_dla34_culane.py --gpus 0
```

**Test / validate**
```bash
python main.py configs/clrnet/clr_dla34_culane.py --test --load_from=culane_dla34.pth --gpus=1
```
Add `--view` to save visualizations to `work_dirs/.../visualization`.

**Demo**
```bash
python main.py configs/clrnet/clr_dla34_culane.py --demo --load_from=<path_to_model.pth> --gpus=1 --view
```

---

## 🛠️ Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=py,pytorch,opencv,git,github,vscode&theme=dark" />

</div>

`PyTorch` · `CUDA` · `MMCV` · `OpenCV` · `Albumentations` · `Scikit-learn` · `Matplotlib` · `ResNet` · `DLA`

---

## 🤝 Acknowledgements

- Built on top of **[CLRNet](https://github.com/Turoad/CLRNet)**: T. Zheng et al., *CLRNet: Cross Layer Refinement Network for Lane Detection*, CVPR 2022.
- Evaluated on **CULane**: X. Pan et al., *Spatial as Deep: Spatial CNN for Traffic Scene Understanding*, AAAI 2018.

## 📖 References

1. X. Pan, J. Shi, P. Luo, X. Wang, X. Tang. *Spatial as Deep: Spatial CNN for Traffic Scene Understanding.* AAAI, 2018.
2. T. Zheng et al. *CLRNet: Cross Layer Refinement Network for Lane Detection.* CVPR, 2022.

---

<div align="center">

**Vishrut Ratnoo** · [vishrutratnoo108@gmail.com](mailto:vishrutratnoo108@gmail.com) · [GitHub](https://github.com/Vishrut108)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%"/>

</div>
