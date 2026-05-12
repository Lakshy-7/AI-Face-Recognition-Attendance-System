import urllib.request
import urllib.error
import json

url = 'http://127.0.0.1:7868/gradio_api/call/v2/get_registered_students'
data = json.dumps({}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req)
    print(resp.read().decode())
except urllib.error.HTTPError as e:
    print('code', e.code)
    print(e.read().decode())
except Exception as e:
    print('error', e)
