# Introduction:

This assignment implements a Auto-Encoder Neural Network based Audio Denoiser. Training speech is a female speaking and training noise is from cafeteria noise, while test speech is a male speaking and test noise is from raining noise.

# Prerequisite:

You need the latest version of `youtube-dl` and `ffmpeg` for Audio downloading. `Sox` is also required for preprocessing(e.g. downsampling and trimming). See their instructions [here](https://ytdl-org.github.io/youtube-dl/download.html) and [here](http://sox.sourceforge.net/) for installation. Apart from that, python packages below are also required
```
scipy
numpy
keras
```

# Run code:

```
python  lab2.py --DOWNLOAD --TRAIN --RECONSTRUCT --PLOT
``` 

A few files will be generated. Most relavant ones are: 

- Audio files: 
    - S_train: clean speech used for training
    - X_train: mixed speech used for training
    - D_train: denoised speech from X_train   
    - S_test: clean speech used for test
    - X_test: mixed speech used for test
    - D_test: denoised speech from X_test

By listening to "X_train.wav" and "D_train.wav", we can see that the training speech is cleaned up pretty well. 

By listening to "X_test.wav" and "D_test.wav", we can see that the test speech was not denoised
as well as the training speech, although the audio indeed become "cleaner" in some way. This means that the denoising network is not so robust and well-generalized because of the diversity in training speech/nosie and test speech/noise. This shows the impact of overfitting. 
    
