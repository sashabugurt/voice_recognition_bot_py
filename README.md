# Voice_Recognition_HSE
## Programming project HSE-2022 (Olga, Maria, Alex and Anastasia)
This is a voice recognition project geared to distinguish between 4 professors of ours: Anton, Oleg, Daniel and Dasha. 
Frontend of this project is a telegram bot (@proga_ficle_bot) where you can upload a .wav/.mp3/.mp4 file or send a voice message to receive a name of the person who is supposingly speaking in this recording.

Our code builds upon SincNet neural network, a CNN for processing raw audio by Mirco Ravanelli, Yoshua Bengio (“Speaker Recognition from raw waveform with SincNet” [Arxiv](https://arxiv.org/abs/1808.00158)).
[The code](https://github.com/sashabugurt/voice_recognition_bot_py/blob/main/bot.py) for telegram bot was made by us, using a token from BotFather. This code is also the main code which integrates all the libraries and packages listed below. 
We also developed [the code](https://github.com/sashabugurt/voice_recognition_bot_py/blob/main/split.py) to split audio files.

## Prerequesits
* Linux/WSL
* Python 3.6
* Conda environment
* Cuda + PyTorch packages 
* NVidia driver (or other GPU)
* Torch package
* Pysoundfile package
* PyDub package
* Numpy library
* [ffmpeg package](https://www.ffmpeg.org/download.html)
* [dnn_models package](https://github.com/mravanelli/SincNet/blob/master/dnn_models.py)
* [speaker_id package](https://github.com/mravanelli/SincNet/blob/master/speaker_id.py)
* [data_io package](https://github.com/mravanelli/SincNet/blob/master/data_io.py)

## Neuro network training
We used 4 .wav files, one for each speaker (duration 1 hour 20 min). Each file was split into smaller .wav files from 3 to 15 seconds. Then the data was manually cleansed from other sounds except professors' voices.
75% of files (580 files) were sent to training, whilst the remaining 25% (202 files) were used for testing. 

The results for the last epoch of training are as follows: 

epoch 80, loss_tr=0.053114 err_tr=0.019141 loss_te=0.105894 err_te=0.034324 err_te_snt=0.000000

## Demo
Here you can watch how our bot works: https://youtu.be/bcUaqwiYWP4

