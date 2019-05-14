import os
import wave
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from python_speech_features  import mfcc
from imblearn.over_sampling import SMOTE

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from scipy.fftpack import fft, ifft
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


def get_wav_path(rootPath):
    """

    :param rootPath: 存放 wav 文件的根目录
    :return: wav 文件路径
    """
    filesPath = []
    for (dirPath, dirNames, fileNames) in os.walk(rootPath):
        for fileName in fileNames:
            if fileName.endswith('.wav') or fileName.endwith('.WAV'):
                filesPath.append(os.sep.join([dirPath, fileName]))
    return filesPath


def get_audio(wav_path):
    """
    读取音频
    :param wav_path: 绝对路径
    :return: 音频，声道，...，采样频率，长度
    """
    f = wave.open(wav_path, 'rb')
    params = f.getparams()

    nChannels, sampWidth, fs, nFrames = params[:4]
    str_audio = f.readframes(nframes=nFrames)
    audio = np.fromstring(str_audio, dtype=np.short)
    return audio, nChannels, sampWidth, fs, nFrames


def my_fft(data):
    """
    对输入的短时语音进行 fft 后，取 5000hz 以下的部分
    :param data: 输入的数据帧
    :return: 频率，功率，fft_data
    """
    fft_data = fft(data)  # 快速傅里叶变换
    # yreal = yy.real  # 获取实数部分
    # yimag = yy.imag  # 获取虚数部分

    yf = abs(fft_data)  # 取模
    yf = yf[range(int(len(data) / 2))]  # 由于对称性，只取一半区间
    yf = yf[:5000]

    xf = np.arange(len(data))  # 频率
    xf = xf[range(int(len(xf) / 2))]  # 取一半区间
    xf = xf[:5000]
    return xf, yf, fft_data[:5000]


def audio_visualization():
    print('audio_visualization!')
    BJ_root = 'E:\python_project\speech_data\ours\sichuan\with-beijing'
    BJ_phrase_root = os.sep.join([BJ_root, 'phrase'])
    BJ_full_root = os.sep.join([BJ_root, 'full'])

    filesPath = get_wav_path(BJ_phrase_root)
    audios = []
    for path in filesPath:
        audio, nChannels, sampWidth, fs, nFrames = get_audio(path)
        audios.append(audio)

    plt.figure(figsize=(10, 8))
    fft_datas = []
    freqs = []
    for i in range(len(audios)):
        freq, abs_fft, fft_data = my_fft(audios[i])
        freqs.append(freq)
        fft_datas.append(fft_data)
        # plt.suptitle('fft fs=16000hz, under 5000hz')
        plt.subplot(4, 5, i+1)
        plt.plot(freq, abs_fft)

    plt.savefig("./outputs/5000hz以下.svg", transparent=True, format='svg')
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in range(len(audios)):
        plt.subplot(4, 5, i+1)
        plt.plot(audios[i])
    plt.savefig("./outputs/原音频.svg", transparent=True, format='svg')
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in range(len(fft_datas)):
        re_BJ = ifft(fft_datas[i])
        plt.subplot(4, 5, i+1)
        plt.plot(re_BJ.real)
    plt.savefig("./outputs/采样后音频.svg", transparent=True, format='svg')
    plt.show()


def get_phrase(path, phrase_len=16000, gap=8000):
    """
    此函数的功能是把一段长时间的音频切割成每段长度为 1s 的小片段
    :param path: 音频路径
    :param phrase_len: 需要剪切的每个片段的长度, 大约长 1s --> 16000 个点
    :param gap: 滑动步长
    :return: 一组 1*16000 长度的片段
    """
    # print('getting a 1s phrase!')
    f = wave.open(f=path, mode='rb')
    params = f.getparams()
    nchannels, sampwidth, fs, nframes = params[:4]
    str_audio = f.readframes(nframes=nframes)
    data = np.fromstring(str_audio, dtype=np.short)

    phrases = []
    # 不包含末尾的点
    for startPoint in range(0, len(data), gap):
        # print(startPoint)
        if startPoint + phrase_len > len(data):
            break
        else:
            phrases.append(data[startPoint:startPoint + phrase_len])
    f.close()
    return phrases


def write_phrase(savePath, data, channels=1, sampwidth=2, framerate=16000):
    """
    输入切割后的 phrase，一次只单独写一个片段 phrase
    :param savePath: 保存路径
    :param data: 要保存的语音 phrase
    :param channels: 声道数
    :param sampwidth:
    :param framerate: 采样频率
    :return:
    """
    data = data.astype(np.short)
    f = wave.open(savePath, 'wb')
    f.setnchannels(channels)
    f.setsampwidth(sampwidth)
    f.setframerate(framerate)
    # turn the data to string
    f.writeframes(data.tostring())
    f.close()
    print('writing success!')


def process_target_data(root_path):
    files_path = get_wav_path(root_path)
    audios = []
    for path in files_path:
        audio, nChannels, sampWidth, fs, nFrames = get_audio(path)
        audios.append(audio)
    fft_datas = []
    re_audios = []
    for i in range(len(audios)):
        freq, abs_fft, fft_data = my_fft(audios[i])
        fft_datas.append(fft_data)
        re_audios.append(ifft(fft_data).real)
    return audios, fft_datas, re_audios


def process_unknown_data(root_path):
    files_path = get_wav_path(root_path)
    all_phrases = []
    for path in files_path:
        phrases = get_phrase(path=path)
        all_phrases.append(phrases)
    return all_phrases


if __name__ == '__main__':
    print('running!')
    BJ_root = 'E:\python_project\speech_data\ours\sichuan\with-beijing'
    BJ_phrase_root = os.sep.join([BJ_root, 'phrase'])
    # BJ_full_root = os.sep.join([BJ_root, 'full'])
    no_BJ_root = 'E:\python_project\speech_data\ours\sichuan\without-beijing'

    _, _, BJ_re_audios = process_target_data(BJ_phrase_root)
    BJ_label = np.zeros([len(BJ_re_audios), 1])
    bj = np.array([var for var in BJ_re_audios])

    # all_noBJ_phrases = process_unknown_data(no_BJ_root)
    files_path = get_wav_path(no_BJ_root)
    no_bj_phrases = get_phrase(path=files_path[0])
    re_phrases = []
    for phrase in no_bj_phrases:
        _, _, fft_data = my_fft(phrase)
        re_phrases.append(ifft(fft_data).real)

    print(len(re_phrases))
    no_BJ_label = np.ones([len(re_phrases), 1])
    no_bj = np.array([var for var in re_phrases])
    print(bj.shape)
    print(no_bj.shape)

    data = np.vstack((bj, no_bj))
    label = np.vstack([BJ_label, no_BJ_label])

    # re_noBJ_audios = []
    # re_phrases = []
    # # 只选一个  多了内存溢出
    # for audio in all_noBJ_phrases:
    #     for phrase in audio:
    #         _, _, fft_data = my_fft(phrase)
    #         re_phrases.append(ifft(fft_data).real)
    #     re_noBJ_audios.append(re_phrases)
    #
    # phrase_count = 0
    #
    # for re_audio in re_noBJ_audios:
    #     for re_phrase in re_audio:
    #         phrase_count += 1
    # no_BJ_label = np.ones([phrase_count, 1])
    # no_bj = []
    # for re_audio in re_noBJ_audios:
    #     for re_phrase in re_audio:
    #         no_bj.append(re_phrase)
    # no_bj = np.array(no_bj)


    print(data.shape)
    print(label.shape)

    smo = SMOTE(random_state=0)
    x_smo, y_smo = smo.fit_sample(data, label)

    x_train, x_test, y_train, y_test = train_test_split(x_smo, y_smo, test_size=0.3)
    xgb_confgs = {
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 18,
        'eval_metric': 'rmse'
    }
    xgbClassifier = xgb.XGBClassifier(**xgb_confgs)
    xgbClassifier.fit(x_train, y_train)

    smo_prediction = xgbClassifier.predict(x_test)
    print('smote 后测试集预测')
    target_names = ['北京时间', '任意片段']
    print(classification_report(y_true=y_test, y_pred=smo_prediction, target_names=target_names))

    ori_prediction = xgbClassifier.predict(data)
    print('smote 前全预测')
    print(classification_report(y_true=label, y_pred=ori_prediction, target_names=target_names))
    ########################################################################################################
    """
    validation
    """
    print('*' * 100)
    files_path = get_wav_path(no_BJ_root)
    no_bj_phrases = get_phrase(path=files_path[1])
    re_phrases = []
    for phrase in no_bj_phrases:
        _, _, fft_data = my_fft(phrase)
        re_phrases.append(ifft(fft_data).real)

    print(len(re_phrases))
    no_BJ_label = np.ones([len(re_phrases), 1])
    no_bj = np.array([var for var in re_phrases])
    print(bj.shape)
    print(no_bj.shape)

    data = np.vstack((bj, no_bj))
    label = np.vstack([BJ_label, no_BJ_label])

    ori_prediction = xgbClassifier.predict(data)
    print('vilidation')
    print(classification_report(y_true=label, y_pred=ori_prediction, target_names=target_names))
    """
    有北京时间的片段测试
    """

