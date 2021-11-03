
from spleeter.separator import Separator

# Use audio loader explicitly for loading audio waveform :
from spleeter.audio.adapter import AudioAdapter

from scipy.io.wavfile import write
from pydub import AudioSegment

separator = Separator('spleeter:2stems')
#separator.separate_to_file('/path/to/audio', '2stem_sep_audio')#separator.separate_to_file('/path/to/audio', '/path/to/output/directory')

audio_loader = AudioAdapter.default()
sample_rate = 22050
#waveform, _ = audio_loader.load('/path/to/audio/file', sample_rate=sample_rate)
waveform, _ = audio_loader.load(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Source Separation\song.ogg', sample_rate=sample_rate)


# Perform the separation :
prediction = separator.separate(waveform)

print(prediction)