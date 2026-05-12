import re, urllib.request, json
html = urllib.request.urlopen('http://127.0.0.1:7868').read().decode()
match = re.search(r'window\.gradio_config = (\{.*?\});', html, re.S)
if match:
    obj = json.loads(match.group(1))
    print('api_prefix', obj.get('api_prefix'))
    print('mode', obj.get('mode'))
    print('enable_queue', obj.get('enable_queue'))
    print('title', obj.get('title'))
    print('has dependencies', 'dependencies' in obj)
    if 'dependencies' in obj:
        print('num deps', len(obj['dependencies']))
        names = [d.get('api_name') for d in obj['dependencies'] if d.get('api_name')]
        print(names)
else:
    print('no match')
