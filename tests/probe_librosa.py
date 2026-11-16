import inspect

import librosa

print("beat_track:", inspect.signature(librosa.beat.beat_track))
print("feature.tempo:", inspect.signature(librosa.feature.tempo))
print("onset_strength:", inspect.signature(librosa.onset.onset_strength))
print("rms:", inspect.signature(librosa.feature.rms))
print("chroma_cqt:", inspect.signature(librosa.feature.chroma_cqt))
print("mfcc:", inspect.signature(librosa.feature.mfcc))
print("segment:", [n for n in dir(librosa.segment) if not n.startswith("_")])
print("beat module:", [n for n in dir(librosa.beat) if not n.startswith("_")])
