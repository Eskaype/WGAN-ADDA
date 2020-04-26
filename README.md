## Domain Adaptation on Medical Images

This repository implements the `WGAN Domain Adaptation` model for single  source and target in *pytorch* framework

The overall model architecture is as shown in the figure:

![][WGAN Model]

[WGAN Model]: (segmentation_pipeline.pdf) "Model architecture"


The code in this repository implements the following features:
* Discriminative domain adaptation (Gradient Reversal Layer)
* Adversarial Domain Adapatation
* WGAN loss + Lipschtiz penalty "*[On Regularization of WGANs](https://arxiv.org/pdf/1709.08894.pdf)*"
* Adversairal Example Agumentation 
* CLAHE: Contrast based augmentations
* WGAN Regularization 
* Learning Rate Scheduler

## Postprocessing methods
* Contour Detection ( OpenCV)
* Ellipsoidal Fitting
* Area Computation 

## Software Requirements
* Python 3.6
* Pytorch v0.4 (needs manual installation from source https://github.com/pytorch/pytorch)
* torchtext
* numpy

One can install the above packages using the requirements file.
```bash
pip install -r requirements.txt
```


## Usage

### Step 1: Preprocessing:
```bash
python datasets/create_dataset.py -o output_folder
```

### Step 2: Train and Evaluate the model:

```bash
python train_gen.py --epochs 100 --batch-size 4 --lr 1e-3
```

```bash
python train_adv.py --epochs 100 --resume 'pretrained_model' --batch-size 4 --lr 1e-4  --gamma
```

```bash
python train_adv_wgan.py --epochs 100 --resume 'pretrained_model' --batch-size 4 --lr 1e-4  --gamma
```

## Dataset

Dataset Statistics included in `data` directory are:

| Dataset                     |Train Set|Dev Set|Test Set|
| --------------------------- |:-------:|------:|-------:|
| REFUGE                      | 300 | 50   | 50    |
| DRISHTI                     | 50 | 25 | 25  |
| ORIGA                       | 650| 50   | 50  |


## Experiments
All the experiments were performed on a modern GE-Force 1080  
Metrics 
* IOU for Cup and Disc
* CDR -> Cup to Disc Ratio 
* AUROC Plot for Glaucoma and Non Glaucoma Detection 

### RESULTS [ Will be published soon]
<!-- [Dataset URL]

| Method                     | IOU Disc | IOU Cup |CDR | 
| ---------------------------|:-----:| :----: |------:| -----:|
| Deeplab V3+                |0.877 | 36.52 |32.5 M | 
| Domain-adversarial (DANN)  |0.89 |69.3 M | 15.5K |
| Adversarial discriminative (ADDA) |0.885 |29.55 | 41.3 M |
| Patch-based adversarial    |0.8944 | 41.8 M | 35.5K |
| WGAN domain adaptation     |0.91| 42.3 M | 52.5K |

<sup>*</sup>1 epoch get completed in around 180 seconds. -->


## Acknowledgements
* Some parts of the code were borrowed from Deeplab V3+ Repo, AdaptSegNet Repo
