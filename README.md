# Humpback Whale Song Classification Project

By Haley Egan

This project was conducted in collaboration with Nan Hauser at the Center for Cetacean Research & Conservation, who generously shared lots of wonder humpback whale audio data.

## Project Overview

This project implements a deep learning pipeline to classify humpback whale vocalizations by geographic location using Convolutional Neural Networks (CNNs). The approach transforms raw audio recordings into spectrogram images, which are then processed using a CNN in Tensorflow for location-based classification.

## Pipeline Architecture

**Audio → Spectrogram → CNN → Location Classification -> Model and Prediction Evaluation**

### Pipeline Components:

1. **Audio Preprocessing**: Raw whale audio files are loaded and preprocessed (cleaned, cropped, etc)
2. **Spectrogram Generation**: Audio signals are converted to time-frequency representations (spectrograms)
3. **CNN Classification**: Convolutional Neural Network analyzes spectrograms to find patterns based on whale location
4. **Output**: Location classification for each audio sample, and prediction of location for new audio samples

## Classification Approach

**Location-Based Classification**: Multi-class classification framework

- Classify each audio sample into one of multiple geographic locations
- Each location represents a distinct class in the classification problem
- Model outputs probability distribution across all possible locations

**Alternative Approaches**:

- Multi-class: One model predicting among all locations (Location A, B, C, D...)
- Binary per location: Multiple binary models (one per location) asking "is this location X or not?"
- Hierarchical: Group locations by region, then classify within regions

## Technical Resources

#### Core Tutorials:

- [TensorFlow Audio Classification Tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio) - Official TensorFlow guide for audio processing
- [CNNs for Audio Classification](https://towardsdatascience.com/cnns-for-audio-classification-6244954665ab/) - Theory and implementation of CNNs for audio
- [MNIST Audio Classification with Spectrograms](https://www.kaggle.com/code/christianlillelund/classify-mnist-audio-using-spectrograms-keras-cnn) - Practical Keras implementation example

#### Advanced Techniques:

- [Custom Audio Classification with TensorFlow](https://towardsdatascience.com/custom-audio-classification-with-tensorflow-af8c16c38689/) - Building custom audio classification models
- [Audio Echo Processing](https://www.kaggle.com/code/naveensgowda/adding-echo-in-audio-and-removing-echo-in-an-audio) - Audio augmentation and noise reduction techniques

## Dataset Information

**Source**: Center for Cetacean Research and Conservation (Nan Hauser's dataset)

#### Audio Format Specifications:

- **Channels**: Stereo audio (2 channels)
- **Structure**: Left and right audio channels are interleaved in single files (alternating left/right channel samples)
- **Implication**: Requires channel separation during preprocessing to access individual left/right audio streams

#### Data Preprocessing Considerations:

- Channel separation may be needed to analyze left vs right audio independently
- Stereo format could provide spatial audio information useful for classification
- File format and sample rate specifications should be documented for consistent processing

## Initial Results

The classification pipeline was tested on the original full-length audio files, as well as shorter 75 second and 30 second clips. 30 second clips proved to be as effective, and occationally better than longer clips in predicting location, and were significantly less computationally expensive, so 30 second clips were used for analysis and development of the pipeline. Further experimentation with audio file lengths is encouraged. 

The segmenting process of audio files can be found in the [SplitAudio_30sec.ipynb](https://github.com/HaleyEgan/Humpback-Whale-Song-Classification/blob/main/SplitAudio.ipynb) and [SplitAudio_75sec.ipynb](https://github.com/HaleyEgan/Humpback-Whale-Song-Classification/blob/main/SplitAudio_75sec.ipynb) notebooks. 

Testing the CNN on the full audio files can be seen at [Spectrogram_to_CNN_FullSong.ipynb](https://github.com/HaleyEgan/Humpback-Whale-Song-Classification/blob/main/Spectrogram_to_CNN_FullSongs.ipynb). 

The full classification notebook with model evaluation and predications on 30 second audio segments can be found in the notebook [HumpbackWhale_SpectrogramCNN_30SecAudioClips.ipynb] (https://github.com/HaleyEgan/Humpback-Whale-Song-Classification/blob/main/HumpbackWhale_SpectrogramCNN_30SecAudioClips.ipynb). 



## Next Steps

1. **Data Exploration**: Analyze audio file formats, sample rates, and duration distributions
2. **Preprocessing Pipeline**: Implement audio loading, channel separation, and spectrogram generation
3. **Model Architecture**: Design CNN architecture optimized for spectrogram classification
4. **Training Strategy**: Develop train/validation/test splits and evaluation metrics
5. **Location Mapping**: Define geographic location categories for classification targets
