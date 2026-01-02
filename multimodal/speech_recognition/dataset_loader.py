
import os
import torch
import torchaudio
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):

    def __init__(self, data_dir, processor, max_samples=None):
        """
        Args:
            data_dir   : Path to LibriSpeech train-clean-100 (or any subset)
            processor  : Wav2Vec2Processor
            max_samples: Optional limit for debugging
        """
        self.data_dir = data_dir
        self.processor = processor

        self.audio_paths = []
        self.transcriptions = []

        self._load_transcripts()

        if max_samples:
            self.audio_paths = self.audio_paths[:max_samples]
            self.transcriptions = self.transcriptions[:max_samples]

        print(f"[DATASET] Loaded {len(self.audio_paths)} audio samples.")

    # ----------------------------------------------------------------------
    # Load all audio + corresponding text files
    # ----------------------------------------------------------------------
    def _load_transcripts(self):
        """
        Walks the directory and loads:
            - *.flac audio files
            - *.trans.txt transcription metadata
        """
        for root, _, files in os.walk(self.data_dir):

            # Find transcription files
            for fname in files:
                if fname.endswith(".trans.txt"):
                    trans_file = os.path.join(root, fname)

                    with open(trans_file, "r") as f:
                        lines = f.readlines()

                    for line in lines:
                       
                        parts = line.strip().split(" ", 1)
                        if len(parts) != 2:
                            continue

                        utt_id, text = parts
                        audio_file = os.path.join(root, utt_id + ".flac")

                        if os.path.exists(audio_file):
                            self.audio_paths.append(audio_file)
                            self.transcriptions.append(text)

    # ----------------------------------------------------------------------
    # Number of items
    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.audio_paths)

    # ----------------------------------------------------------------------
    # Load + process a single audio file
    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        text = self.transcriptions[idx]

        speech, sr = torchaudio.load(audio_path)

        # Resample to 16kHz for Wav2Vec2
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)

        speech = speech.squeeze()  # remove channel dimension

        # Tokenize text
        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids

        # Process audio
        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest"
        )

        inputs["labels"] = torch.tensor(labels)

        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs
