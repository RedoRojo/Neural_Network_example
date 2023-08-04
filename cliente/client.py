import requests
import json

url = 'http://localhost:5000/api/audio'  # URL of the Flask server
file_path = 'arms.mp3'  # path to the audio file

with open(file_path, 'rb') as f:
    files = {'audio': f}
    response = requests.post(url, files=files)

print(response.text)
