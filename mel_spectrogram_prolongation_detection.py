#!/usr/bin/env python3
import csv
import librosa as lr
import matplotlib.pyplot as pp
import numpy as np
import os
import random
import scipy as sp

positives = []
negatives = []

with open('SEP-28k-Extended_clips.csv', 'r') as file:
  for row in csv.DictReader(file):
    path = f"SEP-28k_CLIP/{row['Show']}/{row['EpId']}/{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"
    match int(row['Prolongation']):
      case 0:
        negatives.append(path)
      case 3:
        positives.append(path)

tests = random.sample(negatives, 4 * len(positives))
# tests = positives + negatives
tp = 0
tn = 0
fp = 0
fn = 0

def predict(path, plot=False):
  sr = 16000
  y, sr = lr.load(path=path, sr=sr)

  n_fft = int(0.025 * sr)
  hop_length = int(0.01 * sr)
  n_mels = 40
  S = lr.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

  S_dB = lr.power_to_db(S, ref=np.max)

  rms = lr.feature.rms(y=y, hop_length=hop_length)
  rms[0] = lr.util.normalize(rms[0])

  dissimilarity = lr.util.normalize(np.mean(lr.feature.delta(S_dB, width=5) ** 2, axis=0))

  times = lr.times_like(X=S, sr=sr, hop_length=hop_length)

  peaks = lr.util.peak_pick(x=dissimilarity, pre_max=1, post_max=2, pre_avg=0, post_avg=1, delta=0, wait=0)
  peaks = peaks[dissimilarity[peaks] > 0.05]
  peaks = np.concatenate([[0], peaks, [len(times) - 1]])

  prolongations = []

  for i in range(len(peaks) - 1):
    a = peaks[i]
    b = peaks[i + 1]

    if times[b] - times[a] > 0.25 and np.median(rms[0][a:b]) > 0.1:
      prolongations.append([a, b])

  if plot:
    pp.title(path)
    pp.plot(times, rms[0], color='black')
    pp.plot(times, dissimilarity, color='orange')
    pp.vlines(times[peaks], ymin=0, ymax=1, color='red')

    for a, b in prolongations:
      pp.axvspan(times[a], times[b], alpha=0.5, color='red')
      print(np.median(rms[0][a:b]))

    pp.show()

  return False

for i, path in enumerate(tests):
  print(f"{100 * i / len(tests) : .2f}%")
  prediction = predict(path)

  if path in positives:
    if prediction == True:
      tp += 1

    else:
      fn += 1

  else:
    if prediction == True:
      fp += 1

    else:
      tn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"tp = {tp}")
print(f"tn = {tn}")
print(f"fp = {fp}")
print(f"fn = {fn}")
print(f"precision = {100 * precision:.2f}%")
print(f"recall = {100 * recall:.2f}%")
print(f"f1 = {100 * f1:.2f}%")
print(f"accuracy = {100 * accuracy:.2f}%")
