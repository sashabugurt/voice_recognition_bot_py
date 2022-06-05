from pydub import AudioSegment

f = open("SincNet/data_lists/TIMIT_all.scp")
AudioSegment.ffmpeg = "ffmpeg.exe"
for line in f.readlines():
    line = line.replace("\n", "")
    sound = AudioSegment.from_wav("SincNet/custom_data/" + line)
    sound = sound.set_channels(1)
    sound.export("SincNet/custom_output/" + line, format="wav")
