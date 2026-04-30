# Assignment 3 - Bundle Adjustment


## Requirements

To install requirements :

```bash
cd path\to\03_BundleAdjustment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install matplotlib
pip install torch torchvision torchaudio
### Datasets

This project also requires COLMAP to be installed.

For Windows:

Download colmap-x64-windows-nocuda.zip


Add the bin folder to system PATH

Make sure the command below works:
colmap -h

```

### Training

To optimize the 3D points and camera parameters with bundle adjustment, run:

```bash
python bundle_adjustment.py

```

### Results

Task 1: Bundle Adjustment
![alt text](image.png)

Task 2: COLMAP Reconstruction
![alt text](<2026-04-29 16-54-34.gif>)
 
## Contributing

>📋 This repository is under https://github.com/cleo676767/DIP-Teaching-Assigment-1/edit/main/03_BundleAdjustment

## Acknowledgements

- Thanks to the https://developer.nvidia.com/cuda-11-8-0-download-archive
- Thanks to [face-alignment](https://github.com/1adrianb/face-alignment) by [Adrian Bulat](https://github.com/1adrianb). ](https://github.com/facebookresearch/pytorch3d?tab=contributing-ov-file#)
