from kokoro import KPipeline
import soundfile as sf


# init the pipeline (American English)
pipeline = KPipeline(lang_code='a')

# test case
targets = {
    "prolongation": "The weeeeather is nice.",
    "sound_repetition": "The q-q-quick brown fox.",
    "word_repetition": "I, I, I am here.",
    "block": "Going to...the store.", 
    "interjection": "um... uh... er... I don't know."
}


# Generate the audio files
# O(n^2) complexity
for label, text in targets.items():
    # 'af_bella' or 'af_nicole' clear voice
    generator = pipeline(text, voice='af_bella', speed=1)
    
    for i, (gs, ps, audio) in enumerate(generator):
        filename = f"{label}_{i}.wav"
        sf.write(filename, audio, 24000)
        print(f"Generated: {filename} for text: '{text}'")