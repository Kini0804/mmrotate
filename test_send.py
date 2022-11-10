import requests

filename = 'data/HRSC2016/FullDataSet/AllImages/100000019.bmp'
url = 'http://127.0.0.1:5000'

files = {'file': open(filename, 'rb')}

response = requests.post(url=url, files=files)
print('post_response: {}'.format(response))

file = open('result.jpg', 'wb')
file.write(response.content)
file.close()
