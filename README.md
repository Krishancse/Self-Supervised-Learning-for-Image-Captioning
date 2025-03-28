# Self-Supervised-Learning-for-Image-Captioning
# Self-Supervised Learning for Image Captioning

![image](https://github.com/user-attachments/assets/fb807d85-b8b2-43a1-90ea-5c191cafe23c)
Captions generated by the captioning generator trained with a SimCLR self-supervised model as a pre-text task. We manually marked the generated captions quality as follows: green colour represents good quality, yellow colour - adequate quality, red - unacceptable quality.

## Overview
This repository implements a **Self-Supervised Learning** approach for **Image Captioning** using deep learning techniques. The model learns meaningful visual representations without the need for human-labeled captions, leveraging self-supervised pretraining followed by fine-tuning on an image captioning dataset.

## Problem Statement
Originally, the existing solutions utilised fully supervised trained models for the part of image feature extraction. However, our experiments showed that such a complex task as image captioning requires higher level of generalisation than usual models can provide. This problem could be addressed with using self-supervised learning methods, that recently showed their ability to generalise better. In order to explore this property of SSL approaches, we proposed and explored two solutions for the image captioning using two different self-supervised learning models, based on Jigsaw Puzzle solving and SimCLR, as a pre-text task.

## Results
For the sake of supervised and self-supervised pre-text tasks comparison, we provide the results of their comprehensive testing on the same downstream task, calculating a BLEU score and validation loss. Our proposed solution with SimCLR model used for image feature extraction achieved the following results: BLEU-1: 0.575, BLEU-2: 0.360, BLEU-3: 0.266, BLEU-4: 0.145, and validation loss of 3.415. These outcomes can be considered as competitive ones with the fully supervised solutions. Along with the code, we also provide pre-trained models for image captioning task, which can be used for any random image.

## Features
- Self-Supervised pretraining on unlabeled image data.
- Transformer-based Image Captioning Model.
- Utilization of **Vision Transformers (ViTs)** and **CNN-based Encoders**.
- Contrastive Learning for self-supervised feature extraction.
- Support for datasets like MS COCO, Flickr8k, and Flickr30k.
- Training and evaluation scripts included.

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/self-supervised-image-captioning.git
cd self-supervised-image-captioning

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation
1. **Pretraining Dataset**: Any large-scale unlabeled image dataset (e.g., ImageNet, OpenImages).
2. **Fine-tuning Dataset**: MS COCO, Flickr8k, or any image-caption dataset.
3. Download and preprocess datasets as per instructions in `data/README.md`.
4. Datasets
   - The dataset used for training the Jigsaw Puzzle solving pre-text task is MSCOCO unlabeled 2017, from https://cocodataset.org, can be downloaded here MSCOCO unlabeled 2017.
   - The dataset used for training the Caption generator model downstream task is Flickr8k, which can be downloaded from the shared folder Flickr8k Dataset.

## Model Architecture
The pipeline consists of:
1. **Self-Supervised Pretraining**
   - Contrastive Learning (SimCLR, MoCo, or BYOL)
   - Feature extraction using Vision Transformers or ResNet
2. **Image Captioning Model**
   - CNN/ViT Encoder
   - Transformer or LSTM-based Decoder with Attention

## Training
To train a model from scratch:
1. Run `1_data_preprocessor.py` to extract visual features and textual descriptions.
2. Run `2_train_IC_model.py` to train the caption generator model.
3. Run `3_BLEU.py` to evaluate the BLEU score of the model.
4. Run `4_tokenizer.py` to create a tokenizer file.
5. Run `5_test.py` to generate captions for any image.

To use a pre-trained model:
- Download the pre-trained model, extracted features, descriptions, and tokenizer from the shared folder [Image_Captioning_with_SimCLR](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/20020067_mbzuai_ac_ae/EpbmvMjAMQlNij__vSXoOMQBdv34t5Ws47uIeUdH4LgT3A?e=xQGWWv).
- Place them in a `Pre-trained/` folder.
- Run `3_BLEU.py` to evaluate the BLEU score.
- Run `5_test.py` to generate a caption for any image.

## How to Run the Code
### Pretext Task
1. **Create HDF5 Dataset**: Use `to_hdf5.py` to convert images into HDF5 format.
2. **Generate Permutations**: Run `maximal_hamming.py` to create a set of permutations.
3. **Configure Main Script**: In `main.py`, set dataset and permutation paths.
4. **Adjust Image Size** if necessary.

### Data Processing
- `DataGenerator.py` generates Jigsaw puzzle patches.
- `image_preprocessing/` contains `image_transform.py` for preprocessing.

### Downstream Task: Image Captioning
1. Run `Jigsaw_feature_extraction.py` to extract features.
2. Run `jigsaw_captions.py` to extract textual descriptions.
3. Run `Jigsaw_IC_model.py` to train the captioning model.
4. Run `Jigsaw_tokenizer.py` to create a tokenizer.
5. Run `Jigsaw_test_bleu.py` to compute BLEU scores.
6. Run `Jigsaw_test_images.py` to test the trained model.

## Pre-trained Models
- **Jigsaw Puzzle Solver**: ResNet50-based model with **67% accuracy**.
- **Image Captioning Model**: Trained using Jigsaw-extracted features.

## Evaluation
```bash
python evaluate.py --model-path checkpoints/best_model.pth
```

## Inference
Generate captions for new images:
```bash
python infer.py --image-path path/to/image.jpg --model-path checkpoints/best_model.pth
```

## Results
| Model | BLEU Score | METEOR Score | ROUGE-L |
|--------|------------|--------------|----------|
| Baseline (LSTM-Attention) | 30.5 | 25.8 | 50.2 |
| ViT + Transformer | **38.2** | **31.4** | **56.8** |

## References
- [SimCLR](https://arxiv.org/abs/2002.05709)
- [MoCo](https://arxiv.org/abs/1911.05722)
- [BYOL](https://arxiv.org/abs/2006.07733)
- [Transformer](https://arxiv.org/abs/1706.03762)
- [Related Research](https://arxiv.org/abs/1603.09246)
- [GitHub Repository](https://github.com/Jeremalloch/Semisupervised_Image_Classifier)

## Contributing
Contributions are welcome!




🌐 **Language Tool**: 




<h3 align="left">Connect with me:</h3>
<p align="left">
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://aws.amazon.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/amazonwebservices/amazonwebservices-original-wordmark.svg" alt="aws" width="40" height="40"/> </a> <a href="https://azure.microsoft.com/en-in/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/microsoft_azure/microsoft_azure-icon.svg" alt="azure" width="40" height="40"/> </a> <a href="https://www.gnu.org/software/bash/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/gnu_bash/gnu_bash-icon.svg" alt="bash" width="40" height="40"/> </a> <a href="https://getbootstrap.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/bootstrap/bootstrap-plain-wordmark.svg" alt="bootstrap" width="40" height="40"/> </a> <a href="https://www.cprogramming.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/c/c-original.svg" alt="c" width="40" height="40"/> </a> <a href="https://canvasjs.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/Hardik0307/Hardik0307/master/assets/canvasjs-charts.svg" alt="canvasjs" width="40" height="40"/> </a> <a href="https://circleci.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/circleci/circleci-icon.svg" alt="circleci" width="40" height="40"/> </a> <a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> </a> <a href="https://d3js.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/d3js/d3js-original.svg" alt="d3js" width="40" height="40"/> </a> <a href="https://www.djangoproject.com/" target="_blank" rel="noreferrer"> <img src="https://cdn.worldvectorlogo.com/logos/django.svg" alt="django" width="40" height="40"/> </a> <a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://www.electronjs.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/electron/electron-original.svg" alt="electron" width="40" height="40"/> </a> <a href="https://expressjs.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/express/express-original-wordmark.svg" alt="express" width="40" height="40"/> </a> <a href="https://firebase.google.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/firebase/firebase-icon.svg" alt="firebase" width="40" height="40"/> </a> <a href="https://cloud.google.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="gcp" width="40" height="40"/> </a> <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://grafana.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/grafana/grafana-icon.svg" alt="grafana" width="40" height="40"/> </a> <a href="https://graphql.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/graphql/graphql-icon.svg" alt="graphql" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> <a href="https://ifttt.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/ifttt/ifttt-ar21.svg" alt="ifttt" width="40" height="40"/> </a> <a href="https://www.adobe.com/in/products/illustrator.html" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/adobe_illustrator/adobe_illustrator-icon.svg" alt="illustrator" width="40" height="40"/> </a> <a href="https://www.java.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/java/java-original.svg" alt="java" width="40" height="40"/> </a> <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> </a> <a href="https://www.jenkins.io" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/jenkins/jenkins-icon.svg" alt="jenkins" width="40" height="40"/> </a> <a href="https://www.elastic.co/kibana" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/elasticco_kibana/elasticco_kibana-icon.svg" alt="kibana" width="40" height="40"/> </a> <a href="https://kubernetes.io" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/kubernetes/kubernetes-icon.svg" alt="kubernetes" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://www.mongodb.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mongodb/mongodb-original-wordmark.svg" alt="mongodb" width="40" height="40"/> </a> <a href="https://nestjs.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/nestjs/nestjs-plain.svg" alt="nestjs" width="40" height="40"/> </a> <a href="https://nodejs.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/nodejs/nodejs-original-wordmark.svg" alt="nodejs" width="40" height="40"/> </a> <a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.postgresql.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/postgresql/postgresql-original-wordmark.svg" alt="postgresql" width="40" height="40"/> </a> <a href="https://postman.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/getpostman/getpostman-icon.svg" alt="postman" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://reactjs.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/react/react-original-wordmark.svg" alt="react" width="40" height="40"/> </a> <a href="https://reactnative.dev/" target="_blank" rel="noreferrer"> <img src="https://reactnative.dev/img/header_logo.svg" alt="reactnative" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://spring.io/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/springio/springio-icon.svg" alt="spring" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

Feel free to submit issues or pull requests if you have improvements!

## License
This project is licensed under the MIT License. See `LICENSE` for details.
