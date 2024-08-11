
def del_liness(txt_path, encoding='utf-8'):
    with open(txt_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
    
    # 使用set去重
    unique_lines = set(line.strip() for line in lines)
    
    with open(txt_path, 'w', encoding=encoding) as file:
        for line in unique_lines:
            file.write(line + '\n')

def rank_lines(txt_path, encoding='utf-8'):
    with open(txt_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    # 去掉行末的换行符，并进行排序
    sorted_lines = sorted(line.strip() for line in lines)

    with open(txt_path, 'w', encoding=encoding) as file:
        for line in sorted_lines:
            file.write(line + '\n')


txt_path = r"F:\aaaaaa-xiamen2024-ok\顶盖缺口25Dtxt\头部钢壳ok.txt"

# del_liness(txt_path)
rank_lines(txt_path)
