# Voice_Recognition_HSE
## Programming project HSE-2022 (Olga, Maria, Alex and Anastasia)
This is a voice recognition project geared to distinguish between 4 professors of ours: Anton, Oleg, Daniel and Dasha. 
Frontend of this project is a telegram bot (@proga_ficle_bot) where you can upload a .wav/.mp3/.mp4 file or send a voice message to receive a name of the person who is supposingly speaking in this recording.

**Our code builds upon SincNet neural network, a CNN for processing raw audio by Mirco Ravanelli, Yoshua Bengio (“Speaker Recognition from raw waveform with SincNet” [Arxiv](https://arxiv.org/abs/1808.00158)).**
The code for telegram bot was made by us, using a token from BotFather. 

## Neuro network training
We used 4 .wav files, one for each speaker (duration 1 hour 20 min). Each file was split into smaller .wav files from 3 to 15 seconds. Then the data was manually cleansed from other sounds except professors' voices.
75% of files (580 files) were sent to training, whilst the remaining 25% (202 files) were used for testing. 
