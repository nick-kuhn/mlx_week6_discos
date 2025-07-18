import os
import whisper

def transcribe_audio_files():
    model = whisper.load_model("base")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    audio_files = [
        "(Audio) Week 6 Day 3 Part 1 of 3.wav",
        "(Audio) Week 6 Day 3 part 2 of 3.wav"
    ]
    
    output_files = ["bes1.txt", "bes2.txt"]
    
    for audio_file, output_file in zip(audio_files, output_files):
        audio_path = os.path.join(script_dir, audio_file)
        output_path = os.path.join(script_dir, output_file)
        
        if os.path.exists(audio_path):
            print(f"Transcribing {audio_file}...")
            result = model.transcribe(audio_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            print(f"Transcription saved to {output_file}")
        else:
            print(f"Audio file {audio_file} not found at {audio_path}")

if __name__ == "__main__":
    transcribe_audio_files()