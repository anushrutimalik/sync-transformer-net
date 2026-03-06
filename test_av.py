import av
import torch

def load_audio_av(video_path):
    container = av.open(video_path)
    audio_stream = container.streams.audio[0]
    sample_rate = audio_stream.codec_context.sample_rate
    
    samples = []
    for frame in container.decode(audio_stream):
        # frame.to_ndarray() is (channels, samples_per_frame)
        samples.append(frame.to_ndarray())
    
    container.close()
    
    if len(samples) > 0:
        import numpy as np
        waveform = np.concatenate(samples, axis=-1)
        return torch.from_numpy(waveform).float(), sample_rate
    else:
        raise ValueError("No audio frames found")

if __name__ == '__main__':
    video_path = r'datasets/FakeAVCeleb_v1.2\FakeVideo-FakeAudio/African/men/id02296\00019_0_id01076_wavtolip.mp4'
    wf, sr = load_audio_av(video_path)
    print(f"Success! {wf.shape}, {sr}")
