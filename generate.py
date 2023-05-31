import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


prompt_pre = (
"The following is a conversation between an AI assistant called Doer and a human user called User. "
"The assistant is intelligent, knowledgeable and polite to answer questions of user. "
"Doer由海天瑞声科技股份有限公司（DataOcean.AI）出品。Doer名字来源于公司的首字母简称（DataOcean）。\n\n"
)
prompt_history = "User: {input}\n\nDoer: {output}\n\n"
prompt_post = "User: {input}\n\nDoer: "


def main(
    load_8bit: bool = True,
    base_model: str = "yahma/llama-13b-hf",
    lora_weights: str = "models/llama-13b-lora-alpaca-round-0/checkpoint-6660",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


    def get_prompt(data_point, train_on_inputs=True):
        user_prompt = prompt_pre # 固定开场白
        # 这里面的字段是conversions，而不是input，因为上面的例子的字段是conversations
        conversations = data_point['conversations']
        # 获取多轮对话的轮数
        for i in range(len(conversations) - 1): # 最后一轮对话单独处理，此处不处理
            human = conversations[i]['user']
            assistant = conversations[i]['doer']
            user_prompt += prompt_history.format_map({'input': human, 'output': assistant})
        # 添加最后一轮对话的输入部分
        user_prompt += prompt_post.format_map({'input': conversations[-1]['user']})
        # 根据是训练还是推理，用不同的方式来处理最后一轮对话的回答部分
        if train_on_inputs:
            user_prompt += conversations[-1]['doer']
        return user_prompt

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        # prompt = prompter.generate_prompt(instruction, input)
        data_point = {
            "conversations":[
                {
                    "user": instruction,
                }
            ]
        }
        prompt = get_prompt(data_point, train_on_inputs=False)
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=5.,
            bad_words_ids=tokenizer(['\n\nUser: '], add_special_tokens=False).input_ids, #,'\n\nDoer:'
            **kwargs,
        )
        print(tokenizer(['\n\nUser:','\n\nDoer:'], add_special_tokens=False).input_ids,)
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

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

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)
                    print(decoded_output)
                    print(output[-1])
                    if output[-1] in [tokenizer.eos_token_id]:
                        break
                    yield decoded_output.split("Doer:")[-1].strip()
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    interface = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 DataOcean-DOER",
        description="DOER is a 13B-parameter LLaMA model finetuned to follow instructions.").queue()
    interface.launch(share=True)

if __name__ == "__main__":
    fire.Fire(main)
