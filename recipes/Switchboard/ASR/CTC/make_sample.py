import torchaudio

# sw_4520_B_2953,14.45,1327760,1443360,B,/nfs/data/ldc/LDC2002S09/hub5e_00/english/sw_4520.sph,
# BECAUSE YOU KNOW IF THE TEACHER SAYS DO NOT DO THIS AND THEN KIDS DO IT AND ESPECIALLY THAT AGE THEY WANT TO BE A
# REBEL SO THEY ARE DIFFERENT THEY PROVE THEMSELVES BY YOU KNOW GOING AGAINST THE ESTABLISHMENT TYPE OF THING WHAT DO YOU THINK,sw_4520_B


def audio_pipeline(wav, channel, start, stop):
    # Select a speech segment from the sph file
    # start and end times are already frames.
    # This is done in data preparation stage.
    start = int(start)
    stop = int(stop)
    num_frames = stop - start
    sig, fs = torchaudio.load(wav, num_frames=num_frames, frame_offset=start)
    info = torchaudio.info(wav)

    resampled = torchaudio.transforms.Resample(info.sample_rate, 16000,)(sig)

    resampled = resampled.transpose(0, 1).squeeze(1)
    # Select the proper audio channel of the segment
    if channel == "A":
        resampled = resampled[:, 0]
    else:
        resampled = resampled[:, 1]
    return resampled


if __name__ == "__main__":
    wav = "./sw_4520.sph"
    channel = "B"
    start = 1327760
    stop = 1443360

    sig = audio_pipeline(wav, channel, start, stop)
    print(sig.shape)

    torchaudio.save("example.wav", sig.unsqueeze(0), 16000)
