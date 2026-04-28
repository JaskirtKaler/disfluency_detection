"""
Evaluation script for disfluency detection

Compares ground-truths in a reference csv file and model predictions in a prediction csv file to print out scores for each disfluency type in the prediction csv file:
- number of true positives (tp), true negatives (tn), false positives (fp), false negatives (fn)
- precision, recall, f1-score, accuracy, specificity

The csv files must be in the following format:
- The first line is a header which must contain at least
-- either (a) 'Clip' or (b) a combination of 'Show', 'EpId', 'ClipId' over three columns to specify the name of the audio clip;
-- one or more of the following disfluency types: 'Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'Any'.
- Each line below the header pertains to a different audio clip:
-- The name of the clip (column 'Clip') should not include file extensions like '.wav'.
-- A number (e.g. vote count, probability, 0 for 'no' and 1 for 'yes') must be provided for each disfluency type.

The audio clips and disfluency types in the prediction csv file must be subsets of those in the reference csv file.
"""

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import argparse
import pandas as pd

def load_csv(path, threshold, infer_any=False):
	df = pd.read_csv(path)
	separator = '_'
	if not 'Clip' in df.columns:
		df['Clip'] = df[['Show', 'EpId', 'ClipId']].astype(str).agg(separator.join, axis=1)
	disfluency_types = [dt for dt in ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'Any'] if dt in df.columns]
	df[disfluency_types] = (df[disfluency_types] >= threshold).astype(int)
	if not 'Any' in disfluency_types and infer_any:
		df['Any'] = (df[disfluency_types].sum(axis=1) >= 1).astype(int)
		disfluency_types.append('Any')
	df = df[['Clip'] + disfluency_types]
	return df

def classify(prediction_df, reference_df):
	disfluency_types = set(prediction_df.columns.intersection(reference_df.columns)) - set(['Clip'])
	df = pd.merge(prediction_df, reference_df, on='Clip')
	for d in disfluency_types:
		df[d+'_tp'] = (df[d+'_x'] == 1) * (df[d+'_y'] == 1).astype(int)
		df[d+'_tn'] = (df[d+'_x'] == 0) * (df[d+'_y'] == 0).astype(int)
		df[d+'_fp'] = (df[d+'_x'] == 0) * (df[d+'_y'] == 1).astype(int)
		df[d+'_fn'] = (df[d+'_x'] == 1) * (df[d+'_y'] == 0).astype(int)
	return df, disfluency_types

def count(df, disfluency_type):
	tp = df[disfluency_type + '_tp'].sum()
	tn = df[disfluency_type + '_tn'].sum()
	fp = df[disfluency_type + '_fp'].sum()
	fn = df[disfluency_type + '_fn'].sum()
	return tp, tn, fp, fn

def precision(tp, tn, fp, fn):
	return tp / (tp + fp)

def recall(tp, tn, fp, fn):
	return tp / (tp + fn)

def accuracy(tp, tn, fp, fn):
	return (tp + tn) / (tp + tn + fp + fn)

def specificity(tp, tn, fp, fn):
	return tn / (tn + fp)

def f1(tp, tn, fp, fn):
	p = precision(tp, tn, fp, fn)
	r = recall(tp, tn, fp, fn)
	return 2*p*r / (p+r)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--reference', type=str, help='reference csv file')
	parser.add_argument('--prediction', type=str, help='model prediction csv file')
	parser.add_argument('--reference_threshold', type=float, default=2, help='threshold value to count as "yes" in the reference file')
	parser.add_argument('--prediction_threshold', type=float, default=0.5, help='threshold value to count as "yes" in the prediction file')
	args = parser.parse_args()
	rdf = load_csv(args.reference, args.reference_threshold, infer_any=True)
	pdf = load_csv(args.prediction, args.prediction_threshold)
	df, disfluency_types = classify(rdf, pdf)
	for disfluency in sorted(disfluency_types):
		print('\n#', disfluency)
		tp, tn, fp, fn = count(df, disfluency)
		print('## tp =', tp, ', tn =', tn, ', fp =', fp, ', fn =', fn)
		print('## precision =', round(precision(tp, tn, fp, fn), 3), ', recall =', round(recall(tp, tn, fp, fn), 3), ', f1 =', round(f1(tp, tn, fp, fn), 3), ', accuracy =', round(accuracy(tp, tn, fp, fn), 3), ', specificity =', round(specificity(tp, tn, fp, fn), 3))
