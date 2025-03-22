# Project Title

Image Generation with Deep Convolutional Generative Adversarial Networks (DCGANs).

## Description
This repository consist of two more sub-directories. Notebooks directory contains code for training, evaluating and logs of code execution at different timestamps. These files were run on google colab which provides limited access to compute (maximum time allowed for using compute is 4 hours). To manage training different sessions were used and each session was recorded different file (e.g., AnimeFaceGANs_Run1.ipynb for initial epochs and other files for later training epochs).
This model was deloyped on web app using python flask module which is present in wgan_web directory.
## 🚀 Features
- Implementation of GANs
- Training and Evaluation of DCGANs (WGAN)
- Training process logs
- Unique Anime Character Generation
- Plots of generated images at different epochs

## 🛠️ Installation

To run web app locally, run app.py file in wgan_web directory and open the link on local server. 

```bash
git clone https://github.com/umair627/Assignment2_GAN
cd Assignment2_GAN
