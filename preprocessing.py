# proprocess data files and extract features
import numpy as np
import librosa
# import librosa.display
import glob
import csv
import os.path
# import matplotlib.pyplot as plt

labels = {}
samples = {}
dataFileName = "data.csv"

def get_instrument_name(path):
    return path.split('/')[1]

def generate_label():
    dirs = glob.glob('data/*')
    count = 0
    for dirname in dirs:
        ins = get_instrument_name(dirname)
        labels[ins] = count
        samples[ins] = 0
        count += 1
    return labels

def label_names():
    generate_label()
    return labels.keys()

def trim_silence(sound, silence_threshold=.001):
    max_num = max(sound)
    sound = sound/max_num
    for leading in range(len(sound)):
        if (sound[leading] > silence_threshold):
            break
    for ending in range(len(sound))[::-1]:
        if (sound[ending] > silence_threshold):
            break
    # print(leading, ending)
    return sound[leading:ending+1]

def detect_label(filename):
    ins = get_instrument_name(filename)
    return labels[ins]

def extract_feature():
    sr = 44100
    # window_size = 2048
    # hop_size = int(window_size/2)
    data = []

    #read file
    files = glob.glob('data/*/*.mp3')
    # files = glob.glob('data/viola/viola_D6_025_piano_arco-normal.mp3')
    np.random.shuffle(files)
    count = 0
    for filename in files:
        try:
            note, sr = librosa.load(filename, sr = None)
        except EOFError:
            print("skipped", filename)
            continue
        
        trimmed = trim_silence(note)

        #use mfcc to calculate the audio features
        mfccs = librosa.feature.mfcc(y=trimmed, sr=sr) # shape: time * freq

        # # plot mfcc
        # # print(mfccs.shape)
        # # print(mfccs)
        # # plt.figure(figsize=(10, 4))
        # # librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        # # plt.colorbar()
        # # plt.title('MFCC')
        # # plt.tight_layout()
        # # plt.show()
        
        aver = np.mean(mfccs, axis = 1)
        feature = aver.reshape(20)
        # print(feature)

        samples[get_instrument_name(filename)] += 1

        label = detect_label(filename)
        curline = [" ".join(str(num) for num in feature), label]
        data.append(curline)
        # print(curline)

        count += 1
        if (count % 1000 == 0):
            print(count, "notes processed...")
    
    with open(dataFileName, 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        datawriter.writerows(data)
    print(samples)
    print(sum(samples.values()), "samples in total")
    return data

if not os.path.isfile(dataFileName):
    generate_label()
    print("extracting features...")
    extract_feature()
    print("feature extraction done")
else:
    print("features exist")