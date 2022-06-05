import random

import telebot
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pydub import AudioSegment
from torch.autograd import Variable
from dnn_models import MLP
from dnn_models import SincNet as CNN
from data_io import read_conf, str_to_bool


def get_speaker(te_lst, pt_file):
    np.set_printoptions(threshold=1e6)

    def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp):
        sig_batch = np.zeros([batch_size, wlen])
        lab_batch = np.zeros(batch_size)

        snt_id_arr = np.random.randint(N_snt, size=batch_size)
        rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

        for i in range(batch_size):
            [signal, _] = sf.read(data_folder + wav_lst[snt_id_arr[i]])
            snt_len = signal.shape[0]
            snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
            snt_end = snt_beg + wlen

            sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
            lab_batch[i] = lab_dict[wav_lst[snt_id_arr[i]]]

        inp = Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
        lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

        return inp, lab

    options = read_conf()

    fs = int(options.fs)
    cw_len = int(options.cw_len)
    cw_shift = int(options.cw_shift)

    cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
    cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
    cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
    cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
    cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
    cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
    cnn_act = list(map(str, options.cnn_act.split(',')))
    cnn_drop = list(map(float, options.cnn_drop.split(',')))

    fc_lay = list(map(int, options.fc_lay.split(',')))
    fc_drop = list(map(float, options.fc_drop.split(',')))
    fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
    fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
    fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
    fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
    fc_act = list(map(str, options.fc_act.split(',')))

    class_lay = list(map(int, options.class_lay.split(',')))
    class_drop = list(map(float, options.class_drop.split(',')))
    class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
    class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
    class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
    class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
    class_act = list(map(str, options.class_act.split(',')))

    seed = int(options.seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    wlen = int(fs * cw_len / 1000.00)
    wshift = int(fs * cw_shift / 1000.00)

    Batch_dev = 128
    CNN_arch = {'input_dim': wlen,
                'fs': fs,
                'cnn_N_filt': cnn_N_filt,
                'cnn_len_filt': cnn_len_filt,
                'cnn_max_pool_len': cnn_max_pool_len,
                'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                'cnn_use_laynorm': cnn_use_laynorm,
                'cnn_use_batchnorm': cnn_use_batchnorm,
                'cnn_act': cnn_act,
                'cnn_drop': cnn_drop,
                }

    CNN_net = CNN(CNN_arch)
    CNN_net.cuda()

    DNN1_arch = {'input_dim': CNN_net.out_dim,
                 'fc_lay': fc_lay,
                 'fc_drop': fc_drop,
                 'fc_use_batchnorm': fc_use_batchnorm,
                 'fc_use_laynorm': fc_use_laynorm,
                 'fc_use_laynorm_inp': fc_use_laynorm_inp,
                 'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
                 'fc_act': fc_act,
                 }

    DNN1_net = MLP(DNN1_arch)
    DNN1_net.cuda()

    DNN2_arch = {'input_dim': fc_lay[-1],
                 'fc_lay': class_lay,
                 'fc_drop': class_drop,
                 'fc_use_batchnorm': class_use_batchnorm,
                 'fc_use_laynorm': class_use_laynorm,
                 'fc_use_laynorm_inp': class_use_laynorm_inp,
                 'fc_use_batchnorm_inp': class_use_batchnorm_inp,
                 'fc_act': class_act,
                 }

    DNN2_net = MLP(DNN2_arch)
    DNN2_net.cuda()

    if pt_file != 'none':
        print('LOADING MODEL.')
        checkpoint_load = torch.load(pt_file)
        CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
        DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
        DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

    CNN_net.train()
    DNN1_net.train()
    DNN2_net.train()

    CNN_net.eval()
    DNN1_net.eval()
    DNN2_net.eval()

    with torch.no_grad():
        [signal, _] = sf.read(te_lst)
        signal = torch.from_numpy(signal).float().cuda().contiguous()

        beg_samp = 0
        end_samp = wlen

        N_fr = int((signal.shape[0] - wlen) / (wshift))

        sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
        pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().cuda().contiguous())
        count_fr = 0
        count_fr_tot = 0

        while end_samp < signal.shape[0]:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp + wshift
            end_samp = beg_samp + wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1

            if count_fr == Batch_dev:
                inp = Variable(sig_arr)
                pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
                count_fr = 0
                sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()

        if count_fr > 0:
            inp = Variable(sig_arr[0:count_fr])
            pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))

        [_, best_class] = torch.max(torch.sum(pout, dim=0), 0)
        return best_class.item()


bot = telebot.TeleBot('your_key_from_BotFather_Telegram')
verbs = ['говорит', 'вещает', 'это', 'на записи', 'в головосом', 'говорит']
intro = ['Кажется, что', 'Очевидно, что', 'Наверное,', 'Отвечаю:', 'Стопроцентно', 'Предположу, что', 'Моя догадка, что', 'О,', 'Ага...', 'Это же просто —', 'Элементарно —', 'Похоже, что', 'Думаю,', 'Результат готов:']
speakers = ["Олег", "Даша", "Антон", "Даниил"]


@bot.message_handler(commands=['start'])
def start(message):
    mess = f'Привет, {message.from_user.first_name}! Отправь мне видео или голосовое сообщение'
    bot.send_message(message.chat.id, mess)


@bot.message_handler(content_types=['text'])
def text(message):
    bot.send_message(message.chat.id, 'Пожалуйста, отправь мне видео или голосовое сообщение')


@bot.message_handler(content_types=['video'])
def video(message):
    bot.send_message(message.chat.id, 'Идёт обработка видео...')

    video_info = bot.get_file(message.video.file_id)
    downloaded_file = bot.download_file(video_info.file_path)

    with open(video_info.file_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    output = "".join(video_info.file_path.split(".")[:-1]) + ".mp4"
    sound = AudioSegment.from_file(video_info.file_path)
    sound = sound.set_channels(1)
    sound.export(output, format="wav")

    speaker = get_speaker(output, 'custom_output/model_raw.pkl')
    bot.send_message(message.chat.id, f"{random.choice(intro)} {random.choice(verbs)} {speakers[int(speaker)]}")


@bot.message_handler(content_types=['voice'])
def voice(message):
    bot.send_message(message.chat.id, 'Идёт обработка аудио...')

    audio_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(audio_info.file_path)

    with open(audio_info.file_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    output = "".join(audio_info.file_path.split(".")[:-1]) + ".wav"
    sound = AudioSegment.from_ogg(audio_info.file_path)
    sound.export(output, format="wav")

    speaker = get_speaker(output, 'custom_output/model_raw.pkl')
    bot.send_message(message.chat.id, f"{random.choice(intro)} {random.choice(verbs)} {speakers[int(speaker)]}")


bot.polling(none_stop=True)
