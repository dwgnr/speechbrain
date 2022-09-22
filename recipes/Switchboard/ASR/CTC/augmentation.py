from pathlib import Path

import torch
import torch.nn.functional as F
import random
import speechbrain as sb

from speechbrain.processing.signal_processing import (
    compute_amplitude,
    dB_to_amplitude,
)


class AddBabble(torch.nn.Module):
    """Simulate babble noise by mixing the signals in a batch.

    Arguments
    ---------
    speaker_count : int
        The number of signals to mix with the original signal.
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    mix_prob : float
        The probability that the batch of signals will be
        mixed with babble noise. By default, every signal is mixed.

    Example
    -------
    >>> import pytest
    >>> babbler = AddBabble()
    >>> dataset = ExtendedCSVDataset(
    ...     csvpath='tests/samples/annotation/speech.csv',
    ...     replacements={"data_folder": "tests/samples/single-mic"}
    ... )
    >>> loader = make_dataloader(dataset, batch_size=5)
    >>> speech, lengths = next(iter(loader)).at_position(0)
    >>> noisy = babbler(speech, lengths)
    """

    def __init__(
        self,
        musan_dir,
        speaker_count=3,
        snr_low=0,
        snr_high=0,
        mix_prob=1,
        fixed_snr=False,
    ):
        super().__init__()
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob
        self.fixed_snr = fixed_snr

        self.musan_files = list(Path(musan_dir).rglob("*.wav"))

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            A batch of audio signals to process, with shape `[batch, time]`
        lengths : tensor
            The length of each audio in the batch, with shape `[batch]`.

        Returns
        -------
        Tensor with processed waveforms.
        """

        babbled_waveforms = waveforms.clone()
        # lengths = (lengths * waveforms.shape[1]).unsqueeze(1)
        # batch_size = len(waveforms)
        batch_size = waveforms.shape[0]
        max_len = waveforms.shape[1]
        # transform relative lengths into absolute lengths
        lengths = lengths * max_len
        lengths = lengths.type(torch.LongTensor).to(device=waveforms.device)
        # Don't mix (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return babbled_waveforms

        # Pick an SNR and use it to compute the mixture amplitude factors
        clean_amplitude = compute_amplitude(waveforms, lengths)
        if self.fixed_snr:
            SNR = torch.full(
                (batch_size, 1), self.snr_high, device=waveforms.device
            )
        else:
            SNR = torch.rand(batch_size, 1, device=waveforms.device)
            SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        babbled_waveforms *= 1 - noise_amplitude_factor

        # For each speaker in the batch, select wavefile from MUSAN
        # and add it to the files in the batch
        for i in range(batch_size):
            babbled_waveform = waveforms[i, :].unsqueeze(0)

            wav_len = babbled_waveform.shape[1]
            for j in range(0, self.speaker_count):
                # Load MUSAN waveform
                # babble_waveform, fs = torchaudio.load(random.choice(self.musan_files))
                babble_waveform = sb.dataio.dataio.read_audio(
                    str(random.choice(self.musan_files))
                )
                babble_waveform = babble_waveform.to(waveforms.device)
                babble_len = len(babble_waveform)

                # Padding can be negative. No problemo. It just means we make the MUSAN utterance shorter
                # for some (strange) reason circular padding is only supported by 3D, 4D and 5D tensors
                pad_val = wav_len - babble_len
                babble_waveform = F.pad(
                    input=babble_waveform.unsqueeze(0).unsqueeze(0),
                    pad=(0, pad_val),
                    mode="circular",
                ).squeeze(0)
                babble_len = babble_waveform.shape[1]
                # Rescale
                babble_amplitude = compute_amplitude(
                    babble_waveform, babble_len
                )
                current_new_noise_amplitude = (
                    new_noise_amplitude[i, 0].unsqueeze(0).unsqueeze(0)
                )
                babble_waveform *= current_new_noise_amplitude / (
                    babble_amplitude + 1e-14
                )
                # Add to mixture
                babbled_waveform += babble_waveform
            babbled_waveforms[i, :] = babbled_waveform
        return babbled_waveforms


if __name__ == "__main__":
    import speechbrain as sb

    musan_dir = "/nfs/data/musan/speech"
    wav = "/nfs/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
    sig = sb.dataio.dataio.read_audio(wav).unsqueeze(0)
    sig_len = torch.LongTensor([sig.shape[1]])
    print("sig", sig.shape)
    bab = AddBabble(
        musan_dir,
        speaker_count=3,
        snr_low=0,
        snr_high=15,
        mix_prob=1,
        fixed_snr=True,
    )

    babbled = bab(sig, sig_len).squeeze(0)
    print("babbled", babbled.shape)
    sb.dataio.dataio.write_audio("test_3spk.wav", babbled, 16000)
