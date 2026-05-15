from PyPDF2 import PdfMerger
# 手写签名这种就没有办法保留下来，没什么用
def merge_pdfs(pdf1_path, pdf2_path, output_path):
    # 创建PdfMerger对象
    merger = PdfMerger()
    
    # 将第一个PDF添加到合并器
    merger.append(pdf1_path)
    
    # 将第二个PDF添加到合并器
    merger.append(pdf2_path)
    
    # 将合并后的内容写入新文件
    merger.write(output_path)
    
    # 关闭合并器
    merger.close()
    
    print(f"已成功将 {pdf1_path} 和 {pdf2_path} 合并为 {output_path}")

# 使用示例
if __name__ == "__main__":
    pdf1 = r"D:\ddesktop\新建文件夹\1.pdf"
    pdf2 = r"D:\ddesktop\新建文件夹\2.pdf"
    z = r"D:\ddesktop\新建文件夹\12.pdf"
    merge_pdfs(pdf1, pdf2, z)