# Bird Call Classification using Convolutional Neural Networks

This project uses Convolutional Neural Networks (CNNs) to classify bird species based on their calls using audio data. The goal is to identify 12 common bird species in the Seattle area by converting their sound clips into spectrogram images and training neural networks to recognize them.

## Data
- Source: [Xeno-Canto](https://www.xeno-canto.org/) (crowd-sourced bird sound archive)
- Preprocessing:
  - Audio clips resampled to **22,050 Hz**
  - First 3 seconds extracted
  - Converted to **128x517** spectrograms using 2-second windows
  - Final dataset: 38–630 samples per species

## Neural Network Models

### Binary Classification (American Crow vs. White-crowned Sparrow)
- CNN trained to distinguish between two species.
- Achieved **76.92% test accuracy**.
- Used 2 convolutional layers, ReLU activation, dropout regularization, and binary cross entropy loss.

### Multi-Class Classification (12 Species)
- CNN trained to classify among 12 Seattle-area bird species.
- Achieved **40.98% test accuracy**, significantly better than random chance (≈8.3%).
- Used categorical cross entropy and label binarization for multi-class setup.
