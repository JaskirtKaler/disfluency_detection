import torch
import whisper
import pandas as pd
import os

import sys

# Ensure ffmpeg is in PATH
# py paths
env_bin = os.path.join(sys.prefix, "bin")
if env_bin not in os.environ["PATH"]:
    os.environ["PATH"] = env_bin + os.pathsep + os.environ["PATH"]

# Stay on cpu
device = "cpu"
print(f"🚀 Using Device: {device}")

# Load the model
model = whisper.load_model("base.en", device=device)

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "SEP-28k-Extended_clips.csv")
CLIPS_DIR = os.path.join(SCRIPT_DIR, "../SEP-28k_CLIP") # Base directory from your sidebar


def transcribe_stutter(row):
    # folder structure: Show -> EpId -> Show_EpId_ClipId.wav
    show_name = str(row['Show'])
    # Handle EpId which might be read as float (e.g., 0.0 -> "0")
    try:
        episode_id = str(int(row['EpId']))
    except ValueError:
        episode_id = str(row['EpId'])
        
    clip_id = str(row['ClipId'])
    
    # Filename format: HeStutters_0_0.wav
    filename = f"{show_name}_{episode_id}_{clip_id}.wav"
    
    # Path: ../SEP-28k_CLIP/HeStutters/0/HeStutters_0_0.wav
    file_path = os.path.join(CLIPS_DIR, show_name, episode_id, filename)
    
    if not os.path.exists(file_path):
        return f"MISSING: {file_path}"
    
    # Run the transcription
    try:
        result = model.transcribe(file_path, fp16=False)
        return result['text'].strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

# exe 
try:
    print(f"📂 Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    test_df = df.head(10).copy()
    
    # Quick Diagnostic
    first_row = test_df.iloc[0]
    
    
    show = str(first_row['Show'])
    try:
        ep = str(int(first_row['EpId'])) 
    except:
        ep = str(first_row['EpId'])
    clip = str(first_row['ClipId'])
    fname = f"{show}_{ep}_{clip}.wav"
    
    diag_path = os.path.join(CLIPS_DIR, show, ep, fname)
    print(f"🔍 Checking for file at: {os.path.abspath(diag_path)}")
    
    if os.path.exists(diag_path):
        print("File found!")
    else:
        print("File NOT found during diagnostic check.")

    print(f"Transcribing {len(test_df)} clips on M4 CPU...")
    test_df['whisper_transcript'] = test_df.apply(transcribe_stutter, axis=1)
    
    print("Processed successfully!")
    print(test_df[['Show', 'EpId', 'ClipId', 'whisper_transcript']])
    
    # Optional: Save results
    # output_path = os.path.join(SCRIPT_DIR, "whisper_results_sample.csv")
    # test_df.to_csv(output_path, index=False)
    # print(f"💾 Saved results to: {output_path}")
    
except Exception as e:
    print(f"Error: {e}")
