import gradio as gr
with gr.Blocks() as demo:
    b = gr.Button('Click')
    t = gr.Textbox()
    b.click(lambda: 'x', inputs=[], outputs=[t])
conf = demo.get_config()
print(len(conf.get('dependencies', [])))
print(conf.get('dependencies'))
