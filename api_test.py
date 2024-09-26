import requests
import json


url = 'http://localhost:8000/predict'

text_data = 'ulol manigas ka'

response = requests.post(url, params={'text': text_data})

print(response.json())
