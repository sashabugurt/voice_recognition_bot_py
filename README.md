# Voice_Recognition_HSE
## Programming project HSE-2022 (Olga Vorobieva, Maria Godunova, Alexey Zemtsov and Anastasia Radionova)
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
We used 4 .wav files, one for each speaker (duration 1 hour 20 min).  The data was manually cleansed from other sounds except professors' voices. 

Using our code, we splited files into audio recordings lasting from 3 to 15 seconds in accordance to the requirements of the model. With the help of our code that trains the model (a file with a list of audio recordings by the names of professors was also created there), 75% of files (580 files) were sent to training, whilst the remaining 25% (202 files) were used for testing. 

The results for the last epoch of training are as follows: 

epoch 80, loss_tr=0.053114 err_tr=0.019141 loss_te=0.105894 err_te=0.034324 err_te_snt=0.000000

## Launching the model within the bot

The last action was done automatically by running the model: 

'pip TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp (this line is used in the terminal)', where:
* TIMIT_preparation.py - the code written by the creators of the model
* $TIMIT_FOLDER - the folder where the pre-prepared files are stored
* $OUTPUT_FOLDER - the folder where the audio recordings converted by the model for identification will be stored
* data_lists/TIMIT_all.scp - a list with the names of audio files. 

Next, we moved on to the step of identifying speakers. 

The model was launched with the following line: 

'pip speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg'.

The file in .cfg format was obtained as a result of the previous step. The first file, speaker_id.py , was retrained by us to output the result in the format of indexes and was called 'inference.py'. Now that we have a model in the 'model_raw.pkl' format and other important files, we have moved on to creating a bot. By creating a token for the bot, installing all the necessary libraries and combining the base code to create a bot that accepts video and audio and the code 'inference.py', we have also added several commands to extract audio recordings from videos and voice messages. We created a dictionary of answers of the type "I will assume that this is {the name of the professor}", "It is obvious that {the name of the professor} is on the record". We also created voice and video folders, into which audio files were folded and converted to .wav format.
 

## Demo
Here you can watch how our bot works: https://youtu.be/bcUaqwiYWP4

