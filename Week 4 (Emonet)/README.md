# Week 4: EmoNet
1. [What is EmoNet](#1-what-is-emonet)
2. [Results of Task](#2-results-of-task)
3. [Side Note: Issues with ssh](#3-side-note-issues-with-ssh)

## 1. Overview of Task

EmoNet is the implementation of the [Toisoul 2021](https://www.nature.com/articles/s42256-020-00280-0) paper to perform real-time continuous valence and arousal estimation from images recorded in naturalistic conditions. The implementation was briefly covered in [Week 1 Section 2.2](https://github.com/DJ-Tan/DSO_Internship_JanApr24/tree/main/Week%201%20(Literature%20Review)#22-estimation-of-continuous-valence-and-arousal-levels-from-faces-in-naturalistic-conditions-2021) literature review. 

The aim of this week is to test the model, that has been pretrained using the AffectNet dataset, on the AFEW-VA dataset. However, there may be a slight issue with using EmoNet:
- EmoNet is used for prediction on images, while AFEW-VA is a video-based dataset
- EmoNet will not consider temporal information during the prediction
- Consider finding some pattern across a few frames to estimate the current frame value
- Modify existing model to consider 3D convolutional layers (need to clearly understand the rationale behind current model first)

## 2. Results of Task
The effectiveness of the model on the AFEW-VA dataset was evaluated using the Concordance Correlation Coefficient (CCC) which ranges from -1 to 1, where a score close to 1 indicates high agreement. When the model was tested against the AffectNet dataset it was trained upon, it obtained a CCC score of 0.82 and 0.75 for valence and arousal respectively.

The following table shows the minimum, maximum, and mean VA values when we do a 1-for-1 comparison with the ground truths and predicted values. If we take $\geq$ 0.5 to be a good CCC score (which is a very conservative assumption), then we only have approximately 10 out of 500+ clips that have good VA prediction. Note that the *Combined* column is the CCC score calculated when we combine all clips together instead of considering them separately.

|             | Min    | Max   | Mean  | $\lt$ 0.5 CCC | $\geq$ 0.5 CCC | Combined |
| :---------- | :----: | :---: | :---: | :-----------: | :------------: | :------: |
| **Valence** | -0.763 | 0.703 | 0.022 | 443           | 15             | 0.213    |
| **Arousal** | -0.665 | 0.809 | 0.015 | 515           | 6              | 0.214    |

The following table shows the result after readjusting the predicted values using a moving average window of 5. The rationale is that the model could have made incorrect predictions on a certain frame, so we use the VA values of the surrounding frames to correct the value. While the combined CCC score appears to increase, we see that the range of CCC values (good and bad) gets larger, meaning more variance in the output. This is probably bad since the model is getting less precise.

|             | Min    | Max   | Mean  | $\lt$ 0.5 CCC | $\geq$ 0.5 CCC | Combined |
| :---------- | :----: | :---: | :---: | :-----------: | :------------: | :------: |
| **Valence** | -0.785 | 0.748 | 0.022 | 441           | 17             | 0.218    |
| **Arousal** | -0.788 | 0.851 | 0.016 | 513           | 8              | 0.219    |

In the project specification, the goal is to classify a clip into 1 of the 4 quadrants of the VA chart. The model does this with an accuracy of **57.8%**. Since we have a 25% chance of correctly classifying a clip at random, the model's accuracy could be considered acceptable. It is possible that the metrics used to label clips from AFEW-VA dataset is different from that of the AffectNet dataset, hence accuracy (less specific metric in this case) shows better performance than the CCC score.

## 3. Side Note: Issues with ssh
This section has nothing to do with the EmoNet model, but rather a section to highlight issues faced with attempts to ssh into DSO's AWS server to execute computationally extensive code that cannot be run locally on the issued TechNet.

The standard procedure to set-up is as follows:
1) Generate a pair of ssh keys on our local machine
2) For the first time, send the public key to our supervisor for certification
3) Supervisor returns a certified public key
4) Can now ssh into the instance with `ssh -i [key_filename] [username]@[ip_address]`

The server is set up so that our ssh keys will expire after every 2 weeks, hence recertification has to be done within that period before it expires. This is where the guide is unclear with the instructions.
- Server used for recertification is different from the server with the AWS instance
- Script to regenerate the ssh keys overwrites the old private key
- Implication of overwriting is that we are now unable to ssh into the server to perform the recertification ourselves
- We also lose access to the AWS instance

Since I did not had access to the AWS instance for some period of time, I had to default to some alternatives which were not ideal:
1) Personal laptop with NVIDIA GeForce GTX 1050 (limited memory)
2) NUS SoC computing cluster (cap on processes that can run)
3) NSCC computing cluster (need send job to a queue)

In addition to being unfamiliar with the PyTorch library functions, I was writing code theoretically without knowing if it would actually work. Hence, the progress code-wise during this week was relatively slow. Having reached the 1-month mark in this internship, I hope that I can at least prototype a model that can merge both the audio and visual components to predict VA levels.

