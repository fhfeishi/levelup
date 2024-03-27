# 2024-01-11 21:38
'''
给定一个整数数组 nums 和一个整数目标值 target，
请你在该数组中找出 和为目标值 target  的那 两个 整数，
并返回它们的数组下标。
'''
import copy

class Solution(object):
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            num1 = nums[i]
            for j in range(i+1, len(nums)):
                num2 = nums[j]
                if num1 + num2 == target:
                    print([i,j])
    def twoSum2(self, nums: list, target: int):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
    def isPalindrome(self, y):
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


        # reverse_x = []
        # print(str(y))
        # num = len(str(y))
        # # print(num)
        # for i in range(num):
        #     reverse_x.append((str(y)[i], i))
        # # print(reverse_x)
        # a=reverse_x.sort(key=lambda x:x[1], reverse=True)
        # return num, reverse_x

        # if all[a[i][0] == b[i][0] i for i in range(len(str(y)))]:
        #     return True
        # else:
        #     return False
        # # not use
        # .sort是对list
        # 顺序不就是str(y)吗。。。。。
        # a = str(y).sort(key=lambda x:x.index,  reverse=True)
        # b = str(y).sort(key=lambda x:x.index,  reverse=False)
        # if a == b:
        #     return True
        # else:
        #     return False

'''
.index() 和 .index 是 Python 中用于字符串和列表的方法。这两个方法都用于查找特定元素的索引位置。
# .index()
# 这是字符串和列表的一个方法。当用在字符串上时，它返回指定字符的第一个索引；当用在列表上时，
# 它返回指定元素的第一个索引。如果找不到该字符或元素，它将抛出一个 ValueError。
# #字符串中的用法:  
python`s = "hello"  
print(s.index('e'))  # 输出: 1`  
# # 列表中的用法:   
python`lst = [1, 2, 3, 4, 5]  
print(lst.index(3))  # 输出: 2`  

# .index
# 实际上是 Python 字典的一个方法。
# 它用于查找字典中某个键的第一次出现的位置。如果键不存在，它将抛出一个 KeyError。
d = {'a': 1, 'b': 2, 'c': 3}  
print(d.index('b'))  # 输出: 1`
'''


'''
# 拷贝

# 浅拷贝
a = b

# deepcopy
import copy
a = 

'''

'''
# 回文数，  
reverse_x = []  
for i in range(len(str(y))):   
    reverse_x.append((str(y)[i], i))

# .sort()    
list_yi = []
for i in range(len(str(y))):
    list_yi.append((str(y)[i], i))
# # wrong use
# reverse_yi = list_yi.sort(key=lambda x:x[1], reverse=False)
# return list_yi== reverse_yi
# # list.sort() 方法在排序时会就地修改列表，而不是返回一个新的排序后的列表。
# # 因此，当你执行 reverse_yi = list_yi.sort() 时，reverse_yi 实际上是 None，
# # 这就是为什么你在比较 list_yi == reverse_yi 时得到的结果总是 False。
# wright use
list_yi.sort(key=lambda x:x[1], reverse=False)



# sorted()  



'''

Solution = Solution()
# Solution.twoSum(nums=[2, 7, 11, 34], target=9)
Solution.isPalindrome(-21)
Solution.isPalindrome(21)
Solution.isPalindrome(121)
Solution.isPalindrome(452221)
# z = str(23)
# print(z)






