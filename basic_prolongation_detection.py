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

tests = positives + random.sample(negatives, len(positives))
tp = 0
tn = 0
fp = 0
fn = 0

def predict(path, plot=False):
  y, sr = lr.load(path=path, sr=16000)

  n_fft = int(0.025 * sr)
  hop_length = int(0.010 * sr)

  rms = lr.feature.rms(y=y, hop_length=hop_length)
  rms[0] = lr.util.normalize(rms[0])

  M = lr.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

  # features = np.concatenate((M, lr.feature.delta(M), lr.feature.delta(M, order=2)), axis=0)
  # similarity = 1 - lr.util.normalize([sp.spatial.distance.cosine(features[:, i], features[:, i + 1]) for i in range(features.shape[1] - 1)] + [0])

  similarity = 1 - lr.util.normalize(np.mean(lr.feature.delta(M) ** 2, axis=0))

  similar = similarity > 0.9
  boundaries = np.concatenate(([0], np.where(np.diff(similar))[0], [len(similar) - 1]))
  times = lr.times_like(X=M, sr=sr, hop_length=hop_length)
  prediction = False

  for i in range(len(boundaries) - 1):
    a = boundaries[i]
    b = boundaries[i + 1]

    if times[b] - times[a] > 0.25 and np.mean(rms[0, a:b]) > 0.1:
      prediction = True

      if plot:
        pp.axvspan(times[a], times[b], alpha=0.5, color='red')

  if plot:
    pp.title(path)
    pp.plot(times, rms[0])
    pp.plot(times, similarity)
    pp.vlines(times[boundaries], 0, 1, color='red')
    pp.show()

  return prediction

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
