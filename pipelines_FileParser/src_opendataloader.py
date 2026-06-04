# pipelines_FileParser/src_opendataloader.py  解析pdf

import opendataloader_pdf

# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=[r"D:\ddesktop\26-0325-具身机器头\网申618前\2025湖北省技术创新计划项目申报书-下载0504-------提交-----------------.pdf"],
    output_dir=r"D:\ddesktop\FileParser\output_opendataloader\2025湖北省技术创新计划项目申报书",
    format="markdown,json"
)





