import sys
sys.path.insert(1, r'C:\Users\ewais\Documents\GitHub\tensor-hero\Shared_Functionality\Preprocessing\Preprocessing Functions')
from preprocess_functions import *
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

unprocessed_path = Path(r'X:\Training Data\Unprocessed')

def source_seperate_ogg(ogg_list: list):
    separator = Separator('spleeter:4stems')
    audio_loader = AudioAdapter.default()
    sample_rate = 22050
    range_ = 32767

    for ogg in ogg_list:
        waveform, _ = audio_loader.load(ogg, sample_rate=sample_rate)
        prediction = separator.separate(waveform)
        prediction['other'] = prediction['other']*range_

        save_path = Path(str(ogg).replace('Unprocessed', 'source_separated', 1))
        if not os.path.isdir(save_path.parent):
            os.mkdir(save_path.parent)
        
        print(prediction)
        

        break

ogg_list, _ = get_list_of_ogg_files(unprocessed_path)
source_seperate_ogg(ogg_list)
print(1)