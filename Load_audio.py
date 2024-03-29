# Créateur: Patrice Guyot

import csv
import numpy as np
from scipy.io.wavfile import write

filename = "AUDIO.csv"

samplerate = 11025

audio_list = []
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader, None)  # skip the headers
    for row in csv_reader:
        audio_list.append(int(row[-1]))
audio = np.array(audio_list)
output_file_name = filename.replace(".csv", ".wav")

write(output_file_name, samplerate, audio.astype(np.int16))
print("Audio file written in: ", output_file_name)

