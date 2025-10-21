import torch
import torchaudio

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio

def resample(audio, sr, target_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio 
    waveform = torchaudio.functional.resample(waveform, sr, target_sr) if sr != target_sr else waveform 
    return waveform


# We need to put all the normalization w.r.t waveform and the RIR things here...
class FeatureExtractor(torch.nn.Module):
    def __init__(
        self, 
        in_channels,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

    def _wav2feature(self, waveforms):
        """
        Convert audio waveforms to log-mel filterbank features.
        
        Args:
            waveforms: List of audio waveform tensors
            
        Returns:
            Batch of log-mel filterbank features, padded to match the longest sequence
        """
        features = []
        
        for audio in waveforms:
            # Normalize input audio
            if (audio.ndim == 2) and (audio.shape[0] > 100):
                audio = audio.transpose(1,0)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
                
            audio = normalize_audio(audio, -14.0)
            if audio.shape[0] == 1:
                # For mono audio, duplicate the channel to create stereo
                if self.in_channels == 2:
                    audio = torch.cat((audio, audio), dim=0)
                elif self.in_channels == 4:
                    audio = torch.cat((audio, audio, audio, audio), dim=0)
                elif self.in_channels == 1:
                    audio = audio
                else:
                    raise Exception("Unknowm channel count")
            elif audio.shape[0] == 2:
                if self.in_channels == 1:
                    audio  = audio.mean(axis = 0).unsqueeze(0)
                elif self.in_channels == 2:
                    audio = audio
                else:
                    raise Exception("Unknowm channel count")   
            elif audio.shape[0] == 4:
                if self.in_channels == 1:
                    audio  = audio[0].unsqueeze(0)
                elif self.in_channels == 2:
                    audio = audio[0].unsqueeze(0)
                    audio = torch.cat((audio, audio), dim=0)
                elif self.in_channels == 4:
                    audio = audio                   
            else:
                raise Exception("Unknowm channel count")  
            
            features.append(audio)
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

    def forward(self, x):
        x = self._wav2feature(x).cuda()
        return x

