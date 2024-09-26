import requests

url = "https://fuk.ai/detect-hatespeech/"
headers = {
    "Authorization": "Token 357d4ccff9b0c89e7c67d9a1873ba97bb4738d97330d98060e7b35e645bbf02b"
}
params = {
    "input": "I fucking Like you"
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    try:
        data = response.json()
        print("API Response:", data)
    except requests.exceptions.JSONDecodeError:
        print("Response is not in JSON format.")
        print("Response Content:", response.text)
elif response.status_code == 500:
    print("Server error. Please try again later or contact support.")
    print("Response Content:", response.text)
else:
    print(f"Error {response.status_code}: {response.text}")