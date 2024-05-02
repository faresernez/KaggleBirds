from abc import ABC, abstractmethod
import utils
from librosa.feature import melspectrogram
from librosa.util import normalize

class DataProcessor(ABC):

    def __init__(self,seconds = 5, sr=32000):

        self.seconds = seconds
        self.sr = sr
        self.steps_per_subtrack = seconds*sr

    def loadAudio(self,audio_file):
        chunks , _ = utils.audio_to_chunks(audio_file=audio_file, steps_per_subtrack=self.steps_per_subtrack, sr=self.sr)
        return chunks
    
    @abstractmethod
    def processChunk(self,chunk):
        pass

class melSpectrogram(DataProcessor):

    def __init__(self,seconds, sr, n_mels, hop_length):

        super().__init__(seconds, sr)

        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmax = self.sr / 2
        self.tensorShape = self.processChunk(self.loadAudio('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/unlabeled_soundscapes/460830.ogg')[0]).shape

    def processChunk(self,chunk):
        # return melspectrogram(y = chunk, sr = self.sr, n_mels = self.n_mels, hop_length = self.hop_length, fmax = self.fmax)
        return normalize(melspectrogram(y = chunk, sr = self.sr, n_mels = self.n_mels, hop_length = self.hop_length, fmax = self.fmax))
