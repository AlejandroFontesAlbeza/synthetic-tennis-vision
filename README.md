## Synthetic Tennis Vision Pipeline (Deep Learning + Unreal Engine + Computer Vision)
End-to-end pipeline demonstrating how synthetic data generated in Unreal Engine can be used to train Neural Networks arquitectures to solve a real problem. For this project we applied this workflow for a tennis match segmenting the court lines for homography estimation, and camera pose reconstruction for AR and broadcast applications.

<p align="center">
  <img src="rsc/inference_output.gif" width="45%" />
  <img src="rsc/inference_output.gif" width="45%" />
  <img src="rsc/inference_output.gif" width="45%" />
</p>

---

### Introduction

Camera calibration in sports broadcast and virtual production is traditionally a manual, time-consuming and error-prone process. Accurately estimating camera parameters is essential for rendering augmented reality (AR) graphics aligned with the real world.

This project explores an alternative approach based on synthetic data generation and automatic annotation.

The goal is to demonstrate that:

- Synthetic data generated in a 3D engine can replace manual dataset labeling
- Computer vision models trained on synthetic data can generalize to real-world scenarios
- Camera calibration can be automated using learned visual features

To achieve this, the project implements a full pipeline that:

- Generates synthetic tennis court data using Unreal Engine 5
- Automatically produces perfectly labeled segmentation masks
- Trains a neural network for court line segmentation
- Computes homography and estimates camera pose
- Integrates results back into Unreal Engine for AR visualization

*This is not intended as a production-ready system, but as a technical demonstration of synthetic data pipelines applied to real-world broadcast problems.*

---
### Repository Structure

```
.
├── .github/workflows
│   ├── python_tests.yml
├── data/test_rsc
│   ├── train/
│       ├── images/
│       ├── masks/
│   ├── valid/
│       ├── images/
│       ├── masks/
├── notebooks/
│   ├── synthetic_tennis_camera_pose.ipynb
├── rsc/
├── scripts/ (automation preparation dataset scripts)
├── src/
│   ├── inference/
│   ├── training/
│   ├── unet/
│   ├── utils/
├── tests/
│   ├── test_inference.py
│   ├── test_training.py
├── readmes/
│   ├── data-generation-unreal-engine.md
│   └── computer-vision-pipeline.md
└── .editorconfig
└── .gitignore
└── LICENSE
└── README.md
└── config_inference.py
└── config_training.py
└── dataset_splitter.bat
└── pre_process_dataset.bat
└── pyproject.toml
└── train_model.bat
└── video_inference.bat

```
---
### Documentation

To keep the main README concise and readable, the pipeline is documented in two dedicated sections:

- **Synthetic Data Generation (Unreal Engine)**
→ [README.md](readmes/data-generation-unreal-engine.md)
Covers scene setup, rendering pipeline, and automatic annotation.

- **Camera Pose Estimation Pipeline**
→ [README.md](readmes/computer-vision-pipeline.md)

*Although both parts were developed iteratively, the documentation is structured in a logical order: data generation → model training → inference pipeline.*

---
### Quick Start


```bash
#Clone the repository
git clone https://github.com/AlejandroFontesAlbeza/synthetic-tennis-vision.git
cd synthetic-tennis-vision
```

```bash
#Setup environment
python -m venv venv
venv\scripts\activate # or source venv/Scripts/activate on Linux/macOS
# for CUDA + Windows
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
#for CPU + Windows
pip3 install torch torchvision
pip install -e .
#for other combination I recommend visiting the following link:
```
[Pytorch Documentation](https://pytorch.org/)

- **Train the model**:
    - First of all at ```config_training.py``` you will have to set the corresponding paths to your dataset. If you do not have a dataset. To test de workflow you can set the test_dataset path ```data/test_rsc/train & data/test_rsc/valid```
    - To avoid setting at the cmd always the command python -m src.training.main I create a simple .bat where you can initialize the process and set some parameters depending on your situation.
        ```bash
        train_model.bat --num_classes 10 --lr 1r-4 --batch_size 2 --num_epochs 50
        ```
        ```
        Training...
        100%|██████████| 15/15 [00:20<00:00,  1.34s/it]
        100%|██████████| 10/10 [00:03<00:00,  2.60it/s]
        Epoch 0
        Train Loss: 2.3235, Val Loss: 2.2285
        mIoU %: 0.0318
        Training...
        100%|██████████| 15/15 [00:19<00:00,  1.31s/it]
        100%|██████████| 10/10 [00:03<00:00,  2.55it/s]
        Epoch 1
        Train Loss: 1.3149, Val Loss: 0.3969
        mIoU %: 0.0000
        Training...
        100%|██████████| 15/15 [00:20<00:00,  1.35s/it]
        100%|██████████| 10/10 [00:03<00:00,  2.62it/s]
        Epoch 2
        Train Loss: 0.3337, Val Loss: 0.3333
        mIoU %: 0.0000
        ```
        You can also read which parameters you can stablished setting:
        ```bash
        train_model.bat --help
        ```

- **Run inference**:
    - As the training module, you have a ```config_inference.py``` too. This inference is only available for video inference. If you do not have a video to run the inference, you can use a tennis clip that i leave at rsc/ with the name ```clip1.mp4```
    - To avoid setting at the cmd always the command python -m src.inference.main I create a simple .bat where you can initialize the process and set some parameters depending on your situation.
        ```bash
        video_inference.bat --save_video --show_mask --show_stats
        ```
        As the training module, you can also read the parameters of the script:
        ```bash
        video_inference.bat --help
        ```

## Real-World Application

In sports broadcast and virtual production, accurate camera calibration is essential for rendering augmented reality (AR) graphics aligned with live footage.

Current workflows often rely on manual calibration or specialized hardware, which can be time-consuming and difficult to scale.

This project demonstrates how a combination of synthetic data and computer vision can:

- Automatically recover camera parameters from video
- Reduce manual intervention in calibration workflows
- Enable more scalable and flexible AR pipelines

While the implementation focuses on tennis, the same approach can be applied to other sports and scenarios where scene geometry is known.

<p align="center">
    <img src="rsc/football.gif" width="45%" />
    <img src="rsc/football.gif" width="45%" />

</p>
