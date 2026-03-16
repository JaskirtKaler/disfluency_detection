import json

with open("cnn-transformer.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "absolute_labels_df = df[df[disfluency_cols].eq(3).any(axis=1)].copy()" in source:
            new_source = """df = pd.read_csv('../Training/SEP-28k-Extended_clips.csv')

# 2. Define the columns that represent types of disfluency
disfluency_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']

# Filter the dataframe 
# .eq(3) checks for exactly 3, and .any(axis=1) checks across the columns for each row.
absolute_labels_df = df[df[disfluency_cols].eq(3).any(axis=1)].copy()

# target labels with only 3
def get_absolute_label(row):
    for col in disfluency_cols:
        if row[col] == 3:
            return col
    return None

absolute_labels_df['target_label'] = absolute_labels_df.apply(get_absolute_label, axis=1)

# Now grab fluent data
fluent_df = df[df[disfluency_cols].eq(0).all(axis=1)].copy()
fluent_df['target_label'] = 'Fluent'

# Figure out ball park representation. Let's take the mean of the disfluency class counts
class_counts = absolute_labels_df['target_label'].value_counts()
avg_samples = int(class_counts.mean())

print(f"Disfluency class counts:\\n{class_counts}\\n")
print(f"Total Fluent samples before sampling: {len(fluent_df)}")
print(f"Targeting ~{avg_samples} samples for Fluent class to stay in the ball park.\\n")

if len(fluent_df) >= avg_samples:
    fluent_sampled_df = fluent_df.sample(n=avg_samples, random_state=42).copy()
else:
    fluent_sampled_df = fluent_df.copy()

# Combine Dataframes
absolute_labels_df = pd.concat([absolute_labels_df, fluent_sampled_df], ignore_index=True)

# Shuffle
absolute_labels_df = absolute_labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

# file path map
absolute_labels_df['audio_path'] = (
    'SEP-28k_CLIP/' + 
    absolute_labels_df['Show'] + '/' + 
    absolute_labels_df['EpId'].astype(str) + '/' + 
    absolute_labels_df['Show'] + '_' + 
    absolute_labels_df['EpId'].astype(str) + '_' + 
    absolute_labels_df['ClipId'].astype(str) + '.wav'
)

print(f"Original rows: {len(df)}")
print(f"Filtered rows (absolute agreement + fluent): {len(absolute_labels_df)}\\n")
print(absolute_labels_df['target_label'].value_counts())
print("\\n")
print(absolute_labels_df[['audio_path', 'target_label', 'Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']].head())
"""
            lines = new_source.split('\n')
            new_lines = [l + '\n' for l in lines]
            if new_lines:
                new_lines[-1] = new_lines[-1].strip('\n')
            nb['cells'][i]['source'] = new_lines
            break
            
with open("cnn-transformer.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated cnn-transformer.ipynb")
