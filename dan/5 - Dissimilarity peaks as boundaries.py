#!/usr/bin/env python

import librosa as lr
import numpy as np
import scipy as sp

def predict(path):
  sr = 16000

  y, sr = lr.load(path=path, sr=sr)

  hop_length = int(0.01 * sr)

  rms = lr.feature.rms(y=y, hop_length=hop_length)
  rms_db = lr.amplitude_to_db(S=rms, ref=2e-5)

  times = lr.times_like(X=rms, sr=sr, hop_length=hop_length)

  n_fft = int(0.025 * sr)
  n_mels = 41
  n_mfcc = 13

  M = lr.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)

  dissimilarity = 100 * lr.util.normalize([sp.spatial.distance.cosine(M[:, i], M[:, i + 1]) for i in range(M.shape[1] - 1)] + [0])

  peaks = lr.util.peak_pick(x=dissimilarity, pre_max=1, post_max=2, pre_avg=0, post_avg=1, delta=0, wait=0)
  peaks = peaks[dissimilarity[peaks] > 0.05]
  peaks = np.concatenate([[0], peaks, [len(times) - 1]])

  prolongations = []

  for i in range(len(peaks) - 1):
    a = peaks[i]
    b = peaks[i + 1]

    rms_limit = 55
    duration_limit = 0.25

    if np.median(rms_db[0][a:b]) > rms_limit and times[b] - times[a] > duration_limit:
      prolongations.append([a, b])

  return prolongations

positives = []
negatives = []

with open("SEP-28k-Extended_clips.csv", "r") as file:
  import csv

  for row in csv.DictReader(file):
    path = f"SEP-28k_CLIP/{row['Show']}/{row['EpId']}/{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"

    match int(row["Prolongation"]):
      case 0:
        negatives.append(path)

      case 2 | 3:
        positives.append(path)

tests = positives + negatives

tp = 0
tn = 0
fp = 0
fn = 0

with open("Results 5.csv", "w") as file:
  print("Clip,Prolongation", file=file)

  for i, test in enumerate(tests):
    print(f"{100 * i / len(tests):.0f}%")
    prolongations = predict(test)
    print(f"{test.split('/')[3].replace('.wav', '')},{1 if prolongations else 0}", file=file)

    if test in positives:
      if prolongations:
        tp += 1

      else:
        fn += 1

    else:
      if prolongations:
        fp += 1

      else:
        tn += 1

print(f"tp = {tp}")
print(f"tn = {tn}")
print(f"fp = {fp}")
print(f"fn = {fn}")

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"precision = {100 * precision:.2f}%")
print(f"recall = {100 * recall:.2f}%")
print(f"f1 = {100 * f1:.2f}%")
print(f"accuracy = {100 * accuracy:.2f}%")
