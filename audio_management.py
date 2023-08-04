from pydub import AudioSegment
import pandas as pd 
import os
import math


class AudioUtils:
    def __init__(self):
        self.nose = 1
    def split_mp3(self, 
                  input_file, 
                  output_path, 
                  label,
                  label_id,
                  csv_file, 
                  duration=5000):


        audio = AudioSegment.from_file(input_file, format="mp3")
        total_duration = len(audio)
        num_segments = math.ceil(total_duration / duration)

        num_files  = len([f for f in os.listdir(output_path) 
                           if os.path.isfile(os.path.join(output_path, f))
                        ])
        for i in range(num_segments):

            file_name = f"audio_voz_{num_files+i}.mp3"
            start_time = i * duration
            end_time = start_time + duration
            segment = audio[start_time:end_time]
            segment.export(f"{output_path}/{file_name}", format="mp3")


            data = {"name": [file_name], "classID": [label_id], "class": [label]}
            df = pd.DataFrame(data)
            df.to_csv(csv_file, mode='a', header=False, index=False)

# Usage
# input_file = "./music.mp3"
# output_path = "./audio"
# jose = AudioUtils()
# jose.split_mp3(input_file = input_file,  output_path = output_path, label="jose", label_id="1", csv_file="nose.csv",  duration=5000)
