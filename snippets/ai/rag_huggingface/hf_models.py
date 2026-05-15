from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def download_hf_model(model_name: str) -> None:
    """下载 huggingface-model(LLM)到本地(默认位置 user/.cache/huggingface/models--model_name)"""
    # 下载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",  # 根据你的硬件选择
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    # # # 保存到本地
    # model.save_pretrained("./models/qwen3-1.8b")
    # tokenizer.save_pretrained("./models/qwen3-1.8b")
    print("模型下载完成！")


def create_local_hf_llm(model_name):
    """
    创建本地 Qwen3 chat model，替换原来的 ChatOllama
    """
    
    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    # 创建 transformers pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # 对应 num_predict
        temperature=0.2,     # 对应 temperature
        do_sample=True,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # 使用 HuggingFacePipeline 包装
    llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
        model_kwargs={
            "max_length": 600,  # 对应 num_ctx
            "stop_sequences": ["</result>"],  # 对应 stop 参数
        }
    )
    
    # 使用 ChatHuggingFace 包装成 ChatModel
    chat_model = ChatHuggingFace(
        llm=llm,
        model_id="qwen3-local"
    )
    
    return chat_model



"""
# use_cache_quantization和use_cache_kernel两个参数来控制是否启用KV cache量化
#  use_flash_attn不能与之同时开启

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
     device_map="auto",
     trust_remote_code=True,
     use_cache_quantization=True,
     use_cache_kernel=True,
     use_flash_attn=False
)
"""





