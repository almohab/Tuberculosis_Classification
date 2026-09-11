# Interpretable Deep Learning for Tuberculosis Detection from Chest X-rays

Code and Jupyter notebooks associated with the study:

“Interpretable Deep Learning for Tuberculosis Detection from Chest X-rays: A Comparative Study of Custom and Transfer-Learning Convolutional Neural Networks.”

The repository contains experiments using custom CNNs and transfer-learning architectures, together with Grad-CAM visualization for model interpretability.

## Repository contents

- `notebooks/custom_cnn_20epochs.ipynb` — custom CNN, 20 epochs
- `notebooks/custom_cnn_50epochs.ipynb` — custom CNN, 50 epochs
- `notebooks/custom_cnn_100epochs.ipynb` — custom CNN, 100 epochs
- `notebooks/mobilenetv2_10epochs.ipynb` — MobileNetV2, 10 epochs
- `notebooks/vgg16_20epochs.ipynb` — VGG16, 20 epochs
- `notebooks/vgg16_densenet201_10epochs.ipynb` — VGG16 and DenseNet201, 10 epochs
- `notebooks/inceptionv3_10_20epochs.ipynb` — InceptionV3 experiments
- `notebooks/gradcam_vgg16_tb.ipynb` — Grad-CAM visualization for the VGG16 model

## Dataset

The experiments use chest X-ray images obtained from two publicly available dataset distributions:

1. (Tuberculosis (TB) Chest X-ray Database — Rahman et al.)
   - Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
   - Associated paper: T. Rahman et al., *Reliable Tuberculosis Detection Using Chest X-Ray With Deep Learning, Segmentation and Visualization*, IEEE Access, 2020.
   - DOI: https://doi.org/10.1109/ACCESS.2020.3031384

2. (Dataset of Tuberculosis Chest X-rays Images — Kiran and Jabeen)
   - Mendeley Data, Version 2: https://data.mendeley.com/datasets/8j2g3csprk/2
   - DOI: https://doi.org/10.17632/8j2g3csprk.2

The image files themselves are not included in this repository. Please download the data from the original sources and arrange the working dataset as described below.

## Expected directory structure

```text
tb-xray-deep-learning-classification/
├── notebooks/
├── data/
│   └── TB_Chest_Radiography_Database/
│       ├── Normal/
│       └── Tuberculosis/
├── models/
│   └── tb_vgg16_20epochs.h5
├── requirements.txt
└── README.md
```

The `data/` and `models/` directories are ignored by Git by default so that large files are not accidentally uploaded.

## Installation

Python 3.10 is recommended.

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Running the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open a notebook from the `notebooks/` directory and run its cells in order.

The machine-specific Windows paths from the original experimental notebooks have been replaced with repository-relative paths such as:

```python
base_path = "data/TB_Chest_Radiography_Database"
```

If you store the dataset elsewhere, change `base_path` in the corresponding notebook.

## Model outputs

Some notebooks save trained models locally, for example:

- `mobilenet_tb_model.h5`
- `tb_vgg16_20epochs.h5`
- `tb_cnn_20epochs.h5`
- `tb_cnn7000_model.h5`
- `tb_cnn100_model.h5`
- `InceptionV3_model_name.h5`

These trained model files are not included in this repository.

## Reproducibility note

The notebooks include the original experiment outputs so readers can inspect the reported training and evaluation results. Exact results may vary slightly across TensorFlow versions, hardware, random initialization, and data split execution.

## Software used

Key Python packages include:

- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- OpenCV
- Matplotlib
- seaborn
- Jupyter

See `requirements.txt` for the installation list.

## Data availability

The original chest X-ray images remain available from the public dataset sources listed above. This repository provides the analysis and model-training code only and does not redistribute the image datasets.

## Citation

If you use this repository, please cite the associated manuscript after publication.

## Contact

For questions about the code, please use the GitHub repository Issues page.
