import pandas as pd

file_path = r"D:\Ddesktop\ppt\work\test-xlsx\str2id.xlsx"

df = pd.read_excel(file_path)

# print(df.head(5))

# # 提取'str'列中的后四位字符，并存储在'id'列中
df['id'] = df['str'].apply(lambda x: x[-4:] if isinstance(x, str) else x)

# 提取 'str' 列中的后四位字符，并存储在 'id' 列中
# df['id'] = df['str'].apply(lambda x: x[-4:] if len(x)>4 else x)
# df['id'] = df['str'].astype(str).apply(lambda x: x[-4:] if len(x) > 4 else x)   # 1E22  --> float

# 直接保存修改后的DataFrame回到原Excel文件
df.to_excel(file_path, index=False)
