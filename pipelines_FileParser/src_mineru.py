# pipelines_FileParser/src_mineru.py  解析pdf

import os
import subprocess


input_file = r"D:\ddesktop\26-0325-具身机器头\网申618前\2025湖北省技术创新计划项目申报书-下载0504-------提交-----------------.pdf"
output_file = r"D:\ddesktop\FileParser\output_mineru\2025湖北省技术创新计划项目申报书"
mineru_bin = r"D:\environment\miniconda\envs\enva\Scripts\mineru.exe"
command = [mineru_bin, "-p", input_file, "-o", output_file, "-b", "pipeline", "-m", "auto"]

env = os.environ.copy()

## huggingface
# env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
env.setdefault("HF_ENDPOINT", "https://huggingface.co")

env.setdefault("HF_HOME", r"E:\local_models\huggingface\cache")
env.setdefault("MODELSCOPE_CACHE", r"E:\local_models\modelscope")

## modelscope
# env.setdefault("MINERU_MODEL_SOURCE", "modelscope")

## cpu
env.setdefault("MINERU_DEVICE_MODE", "cpu")

print("MINERU_MODEL_SOURCE=", env.get("MINERU_MODEL_SOURCE"))
print("HF_ENDPOINT=", env.get("HF_ENDPOINT"))
print("HF_HOME=", env.get("HF_HOME"))
print("MODELSCOPE_CACHE=", env.get("MODELSCOPE_CACHE"))
print("MINERU_DEVICE_MODE=", env.get("MINERU_DEVICE_MODE"))

subprocess.run(command, check=True, env=env)


