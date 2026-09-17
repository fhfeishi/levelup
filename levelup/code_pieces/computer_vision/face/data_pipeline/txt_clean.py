# txt重复行处理
from pathlib import Path

def deduplicate_bbox_file(input_file, output_file=None):
    """
    去除bbox.txt中的重复行
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（默认覆盖原文件）
    """
    if output_file is None:
        output_file = input_file
    
    # 读取所有行并去重（保持顺序）
    seen = set()
    unique_lines = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
    
    # 写回文件
    with open(output_file, 'w') as f:
        for line in unique_lines:
            f.write(line + '\n')
    
    total_lines = len(unique_lines) + (len(seen) - len(unique_lines)) * 2  # 估算原始行数
    duplicates = total_lines - len(unique_lines)
    
    print(f"{'='*60}")
    print(f"✓ Deduplication completed!")
    print(f"  Original lines: ~{total_lines}")
    print(f"  Unique lines: {len(unique_lines)}")
    print(f"  Removed duplicates: ~{duplicates}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")

# 使用方法
if __name__ == '__main__':
    # 方案1：直接覆盖原文件
    deduplicate_bbox_file(r'D:\ddesktop\robotss\codespace\bbox.txt')
    
    # 方案2：保存到新文件
    # deduplicate_bbox_file('bbox.txt', 'bbox_clean.txt')



