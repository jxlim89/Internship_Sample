# Week 5: Vision Transformer
1. [What is the Vision Transformer?](#1-what-is-the-vision-transformer)
2. [Using the Model](#2-using-the-model)


## 1. What is the Vision Transformer?
The Vision Transformer (ViT) is a neural network model that adapts the Transformer architecture used for Natural Language Processing (NLP) tasks to perform image classification. The model was first introduced in 2021 by Dosovitskiy et al. ([link](https://arxiv.org/pdf/2010.11929.pdf)). The goal of the research was to show that Transformers can be used in place of Convolutional Neural Networks (CNNs), consuming lesser resources while achieving good results.

The model can be described in 3 components:
1. Patch + Position Embedding
    - Split an image into multiple sub-images (patches)
    - Flatten the patches into a linear projection (2D array)
2. Transformer Encoder
    - Captures the local and global relationships within the image
3. MLP Classification Head
    - Attached to the final layer of the encoder
    - Predict the class label(s)

<p align="middle">
  <img src="Images/ViT Model.png"/>
  <br>Figure 1: Model overview for ViT Transformer
</p>

The model was trained on ILSVRC-2012 ImageNet, ImageNet-21k, and JFT datasets. Over various popular benchmarks, the model had a mean accuracy ranging from 72% to 99%. 

## 2. Using the Model
In Week 4, the AFEW-VA dataset was tested against the EmoNet model. Using a similar dataset, the goal is to see if the ViT can exhibit better performance. From this, either the HourGlass model from EmoNet, or the Transformer architecture from ViT will be adapted into our final model.

There will be 3 different variations of the ViT model tested:
1. Directly using the pretrained ViT model (base)
2. Pretraining on FER2013 dataset
3. Pretraining on FER2013 dataset with noise augmentation

The characteristics of the FER2013 dataset are (also mentioned in [Week 1](https://github.com/DJ-Tan/DSO_Internship_JanApr24/tree/main/Week%201%20(Literature%20Review)#3-datasets-to-consider)):
- Dataset of over 30,000 facial images
- 48x48 pixels grayscale-images
- Labelled with 1 of 7 emotions: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

It should be noted that the FER2013 dataset uses a label with 7 classes, while the AFEW-VA dataset stores VA values instead. To make it consistent for comparison, the 7 classes were sorted into the four quadrants of the 2-D VA chart as follows:
- Angry, Disgust, Fear -> Quadrant 1 (Low Valence, High Arousal)
- Happy, Surprise -> Quadrant 2 (High Valence, High Arousal)
- Sad -> Quadrant 3 (Low Valence, Low Arousal)
- Neutral -> Quadrant 4 (High Valence, Low Arousal)

Similarly, the VA values of the AFEW-VA dataset were sorted into these 4 quadrants.

In the 3rd variation of the model, noise augmentation was applied on the test data. The goal is to make the model more resilient to recognizing noisy facial images in real-world scenarios. It is predicted that testing the noise augmented model on the AFEW-VA dataset should improve accuracy, since movie scenes tend to be darker and grainy.

The following noise augmentations were used:
1. Gaussian noise
    - Type of noise sampled from a Gaussian distribution
    - Simulates sensor/electrical noise
2. Poisson noise
    - Type of noise sampled from a Poisson distribution
    - Simulates noise from a low-light setting
3. Salt and Pepper noise
    - Type of noise characterized by random bright and dark pixels
    - Simulates noise caused by corrupted pixels and/or faulty sensors

<p align="middle">
  <img src="Images/Noise Augmentation.png"/>
  <br>Figure 2: Noise augmentation of test images (from left to right): Original, Gaussian, Poisson, Salt and Pepper
</p>

The following are the accuracy results of the 3 models:
| Model   | Evaluation Accuracy | Test Accuracy |
| :-----: | :-----------------: | :-----------: |
| Model 1 | NIL                 | 21.84%        |
| Model 2 | 70.25%              | 35.34%        |
| Model 3 | 94.14%              | 33.84%        |

1. Model 1 (base)
    - Performed worse than random (21.84% < 25.00%)
    - Model inputs could have been processed differently in pre-training
2. Model 2
    - Best performing model out of the 3
    - Various instances where an entire clip had an incorrect prediction
    - Discrepancies in comparing emotion labels and VA values?
3. Model 3
    - Noise augmentation may have exacerbated issue of class imbalance
    - There were twice as many classes with high arousal as compared to low arousal
    - As a result, the model could probably predict better for inputs that are high arousal, as compared to when they are low arousal
    - If the AFEW-VA dataset had more low arousal clips, the model may not perform as well then

Note that the ViT performs image classification without the temporal component, which is not the most ideal for video classification since not all the data is not fully utilised for prediction. Alternatively if we were to consider proceeding forward with this model, we can consider testing various other modules for the transformer layer, or redefining a new MLP classification head after the encoder. 

For Week 6, the goal is to continue work on making the VideoMAE model more efficient, to avoid potentially crashing the AWS server again.