
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2Inference:

    def __init__(self, model_path):
        print("[INFO] Loading Wav2Vec2 model for inference...")

        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)

        self.model.eval()

    # ----------------------------------------------------------------------
    # Resample audio to 16kHz mono
    # ----------------------------------------------------------------------
    def load_audio(self, file_path):
        speech, sr = torchaudio.load(file_path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)

        speech = speech.squeeze()

        return speech, 16000

    # ----------------------------------------------------------------------
    # Run inference (speech → text)
    # ----------------------------------------------------------------------
    def transcribe(self, audio_path):
        speech, sr = self.load_audio(audio_path)

        inputs = self.processor(
            speech,
            sampling_rate=sr,
            return_tensors="pt",
            padding="longest"
        )

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(pred_ids[0])

        return transcription


# ----------------------------------------------------------------------
# Standalone Use-Case 
# ----------------------------------------------------------------------
if __name__ == "__main__":

    model_path = "./my_wav2vec2_model"  # CHANGE TO YOUR TRAINED MODEL
    audio_file = "sample.wav"

    infer = Wav2Vec2Inference(model_path)

    print("\n[TRANSCRIPTION]")
    result = infer.transcribe(audio_file)
    print(result)
