#!/usr/bin/env python

import librosa as lr
import numpy as np
import os
import scipy as sp

if not os.path.exists("speakers_average_speaking_rate.csv"):
  import collections
  import csv

  def get_peaks(path):
    sr = 16000

    y, sr = lr.load(path=path, sr=sr)

    hop_length = int(0.01 * sr)

    rms = lr.feature.rms(y=y, hop_length=hop_length)
    rms_db = lr.amplitude_to_db(S=rms, ref=2e-5)

    peaks = lr.util.peak_pick(x=rms_db[0], pre_max=1, post_max=2, pre_avg=0, post_avg=1, delta=0, wait=0)
    return peaks

  speakers = {}
  total_peaks = collections.defaultdict(int)
  total_duration = collections.defaultdict(int)

  with open("SEP-28k-Extended_clips.csv", "r") as file:
    for row in csv.DictReader(file):
      path = f"SEP-28k_CLIP/{row['Show']}/{row['EpId']}/{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"
      speaker = f"{row["Show"]}_{row["EpId"]}_{row["is_probably_host"]}"
      speakers[path] = speaker
      total_peaks[speaker] += len(get_peaks(path))
      total_duration[speaker] += 3

  speaking_rate = {speaker: total_peaks[speaker] / total_duration[speaker] for speaker in total_peaks}

  with open("speakers_average_speaking_rate.csv", "w") as file:
    for path in speakers:
      print(f"{path},{speaking_rate[speakers[path]]}", file=file)

else:
  speaking_rate = {}

  with open("speakers_average_speaking_rate.csv", "r") as file:
    for line in file:
      path, rate = line.strip().split(",")
      rate = float(rate)
      speaking_rate[path] = rate / 2

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

  prolongations = []
  i = 0

  for j in range(len(times)):
    rms_limit = 55
    dissimilarity_limit = 10

    if rms_db[0, j] > rms_limit and dissimilarity[j] < dissimilarity_limit:
      continue

    duration_limit = 1.2 / speaking_rate[path]

    if times[j] - times[i] > duration_limit:
      prolongations.append([i, j])

    i = j

  return len(prolongations) > 0

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

with open("Results 1.csv", "w") as file:
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
