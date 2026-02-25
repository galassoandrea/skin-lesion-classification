# 🔬 Skin Lesion Classification with Transformer-Based Models

## 📖 Overview

The objective of this project is to fine-tune a transformer-based pretrained model for the task of Skin Lesion Classification in medical images.

## 🛠️ Technical Details

### 💾 Dataset

The dataset used in this work is the Skin Cancer MNIST: HAM10000, which contains 10015 dermatoscopic images from different populations, including a representative collection of all important diagnostic categories in the realm of pigmented lesions:

- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
- basal cell carcinoma (bcc)
- benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses bkl)
- dermatofibroma (df)
- melanoma (mel)
- melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

The dataset includes lesions with multiple images, which can be tracked by the lesion_id column within the HAM10000_metadata file.

### 🤖 Model

The model I choose to fine-tune is DeiT (Distilled data-efficient Image Transformer), specifically, the variant deit-base-distilled-patch16-224.
This model is a distilled Vision Transformer (ViT), which uses a distillation token, to effectively learn from a teacher (CNN) during both pre-training and fine-tuning.
It was pre-trained and fine-tuned on ImageNet-1k (1 million images, 1,000 classes) at resolution 224x224.
The reason why I chose it instead of ViT is that DeiT allows to achieve state-of-the-art transformer performance, even though it was specifically designed to be trained on smaller datasets like ImageNet-1k (which is more similar in dimensions to the HAM10000 dataset, than the massive JFT-300M dataset that ViT required).

## 👨🏻‍💻 Implementation

The project consists in a series of steps:

1. Download the HAM10000 dataset from kaggle.

2. Load the dataset and perform an Exploratory Data Analysis (EDA).

3. Preprocess the dataset based on the results of the EDA and split it into training, validation and test sets.

4. Define loss function, optimizer and evaluation metrics.

5. Load the pretrained model, add a custom a classification head and fine-tune it for 20 epochs on the task.

6. Evaluate the model performance after training.

Entering more into the details:

- Since the dataset is highly imbalanced, a stratified split is performed to ensure that the same portion of classes (diagnostic categories) are included in each subset. Additionally, class weights are computed and passed inside the loss function during training: this allows to penalize more classification errors made on less frequent classes.

- Considering that there are cases where a lesion is represented by more than one image, the dataset is splitted in such a way that all the images representing the same lesion are put inside the same subset.

- The fine-tuning technique I decided to apply is full-model fine-tuning (i.e. fine-tuning both the backbone and the new classification head): the reason for this is that medical images often require models pretrained on generic images (like the ones from ImageNet) to adjust their internal "filters" to "see" medical features (like pigment networks or globules) that don't exist in those generic images.

## 📊 Results

Below are reported the results obtained during the model evaluation on the test set, along with the confusion matrix:

- Accuracy = 0.7863
- F1 = 0.6546
- AUC = 0.9512

![Confusion Matrix](outputs/confusion_matrix.png)

Apparently, the fine-tuned model achieves a very high AUC but a significantly lower F1-score suggesting that, while it can in general predict the correct class, it struggles with rarer classes.
In other words, the accuracy is highly carried by the nv (Melanocytic nevi) class, which represents more than half of the dataset.

Additionally, from the confusion matrix it is possible to see that in some cases, the misprediction can fall in less frequent categories (look at akiec-bcc or akiec-mel). In part I expected this, since medical datasets can be very complex and during the EDA step it emerged that images belonging to different diagnostic categories, can be very similar between them, making it hard to distinguish classes even for humans.

Anyway, please keep in mind that **the current version is just the first version of the project**, which will likely see further updates and improvements in the next days/weeks.
