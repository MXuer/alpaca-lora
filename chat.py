from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import gradio as gr
import mdtex2html
import torch
import sys
import transformers
from peft import PeftModel
from utils.callbacks import Iteratorize, Stream


"""Override Chatbot.postprocess"""

device = "cuda"

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


# get prompt template
prompt_pre = (
"The following is a conversation between an AI assistant called Doer and a human user called User. "
"The assistant is intelligent, knowledgeable and polite to answer questions of user. "
"Doer由海天瑞声科技股份有限公司（DataOcean.AI）出品。Doer名字来源于公司的首字母简称（DataOcean）。\n\n"
)
prompt_history = "User: {input}\n\nDoer: {output}\n\n"
prompt_post = "User: {input}\n\nDoer: "


def get_prompt(history, input):
    user_prompt = prompt_pre # 固定开场白
    # 这里面的字段是conversions，而不是input，因为上面的例子的字段是conversations
    # 获取多轮对话的轮数
    for i in range(len(history)): # 最后一轮对话单独处理，此处不处理
        human = history[i][0]
        assistant = history[i][1]
        user_prompt += prompt_history.format_map({'input': human, 'output': assistant})
    # 添加最后一轮对话的输入部分
    user_prompt += prompt_post.format_map({'input': input})
    return user_prompt

# model load
base_model = "yahma/llama-13b-hf"
lora_model = "models/llama-13b-lora-alpaca-round-0/checkpoint-6840"

model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    lora_model,
    torch_dtype=torch.float16,
)

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

tokenizer = LlamaTokenizer.from_pretrained(base_model)

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, repetition_penalty, history):
    chatbot.append((parse_text(input), ""))
    # get prompt
    prompt = get_prompt(history, input)
    print(prompt)
    inputs = tokenizer(prompt, 
                       return_tensors="pt", 
                       max_length=max_length,
                       truncation=True)
    input_ids = inputs["input_ids"].to(device)
    # configuration for generation
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        num_beams=4,
        repetition_penalty=float(repetition_penalty),
        bad_words_ids=tokenizer(['\n\nUser: '], add_special_tokens=False).input_ids, #,'\n\nDoer:'
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_length,
    }

    def generate_with_callback(callback=None, **kwargs):
        kwargs.setdefault(
            "stopping_criteria", transformers.StoppingCriteriaList()
        )
        kwargs["stopping_criteria"].append(
            Stream(callback_func=callback)
        )
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(
            generate_with_callback, kwargs, callback=None
        )
    history.append((input, ""))
    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            # new_tokens = len(output) - len(input_ids[0])
            decoded_output = tokenizer.decode(output)
            print(decoded_output)
            print(output[-1])
            if output[-1] in [tokenizer.eos_token_id]:
                break
            history[-1] = chatbot[-1]
            latest_answer = decoded_output.split("Doer:")[-1].strip()
            history.append((input, latest_answer))
            chatbot[-1] = (parse_text(input), parse_text(latest_answer))
            yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">DataOcean.AI</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            repetition_penalty = gr.Slider(1, 20, value=10, step=.5, label="RepetitionPenalty", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, repetition_penalty, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True)
