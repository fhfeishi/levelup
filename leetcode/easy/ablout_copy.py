import copy

a = 5
b = copy.deepcopy(a)

a = 10  # 改变a的值，b的值不会改变
print(b)  # 输出：5

a = 5
b = a  # 这是浅拷贝，如果a是一个复杂的数据结构（如列表、字典等），b和a会指向同一个对象

a = 10  # 改变a的值，b的值不会改变
print(b)  # 输出：5

import copy

a = [1, 2, 3]
b = copy.copy(a)  # 浅拷贝，b和a指向不同的对象，但内容相同

a[0] = 10  # 改变a的值，b的值不会改变
print(b)  # 输出：[1, 2, 3]
# 理解上面的逻辑

# 下面的why
def isPalindrome(y):
    # 1 #
    # str_a = str(y)  # 将整数转换为字符串
    # a_list = [(int(digit), i) for i, digit in enumerate(str_a)]  # 存储每个数字的索引和值
    # sorted_list = sorted(a_list, key=lambda x: x[1], reverse=True)  # 根据索引对列表进行排序
    # return a_list == sorted_list  # 判断两个列表是否相等

    # # 2# ok    = copy.deepcopy()
    # list_yi = [(str(y)[i], i) for i in range(len(str(y)))]
    # list_yi_2 = copy.deepcopy(list_yi)
    # reverse_yi = sorted(list_yi_2, key=lambda x: x[1], reverse=True)
    # print(list_yi_2)
    # print(reverse_yi)

    # # 3 #  all True
    # list_yi = []
    # for i in range(len(str(y))):
    #     list_yi.append((str(y)[i], i))
    # list_base = (list_yi,)
    # list_yi.sort(key=lambda x: x[1], reverse=True)
    # print(list_yi)
    # print(list_base[0])

    # 4#  --ok
    str_a = str(y)  # 将整数转换为字符串
    a_list = [(digit, i) for i, digit in enumerate(str_a)]  # 存储每个数字的索引和值
    sorted_list = sorted(a_list, key=lambda x: x[1], reverse=True)  # 根据索引对列表进行排序
    a_list = [digit[0] for digit in a_list]
    sorted_list = [digit[0] for digit in sorted_list]
    print(a_list)
    print(sorted_list)

# 还有之前
records= [
    ({'bclass': 'z', 'id':None, 'num':4, 'uu':True}),
    ({'a':1, 'id':None, 'num':3})
]
for rec in records:
    print(rec)
    # # 这样的方法，如果没有这个 'bclass'的话无论如何都会报错keyerror
    # if 'z' in rec['bclass']:
    # if rec['bclass'] is 'z':

    # 应该这样
    # 如果 'bclass' 键不存在，get() 方法将返回 'z'
    # if 'z' == rec.get('bclass', 'z'):   # # ‘z’只是说得到一个值，没什么用的值
    if 'z' == rec.get('bclass'):
        rec['uu'] = None
        rec['id'] = 'a'
        # 成功改变了原本records中的字典
        # 为什么
    else:
        pass
print(records)
# 可以看到一个





