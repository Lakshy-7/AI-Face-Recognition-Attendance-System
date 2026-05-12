import gradio as gr
from app import build_app

demo, theme, css = build_app()
print('api_info exists:', hasattr(demo, 'api_info'))
if hasattr(demo, 'api_info'):
    print('api_info keys:', list(demo.api_info.keys()))
    print('named_endpoints count:', len(demo.api_info.get('named_endpoints', {})))
    for k in list(demo.api_info.get('named_endpoints', {}).keys())[:50]:
        print('endpoint key:', repr(k))
print('dependencies count:', len(demo.get_config().get('dependencies', [])))
for dep in demo.get_config().get('dependencies', []):
    print('dep:', dep.get('api_name'), dep.get('targets'))
