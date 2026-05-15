

# def TwoSum(num_lst, target, nums=2):
#     for i, x in enumerate(num_lst):
#         for j, y in enumerate(num_lst):
#             if i != j and (x+y) == target:
#                 a = (i,j)
#                 break 
#         break    
#     return a       

def TwoSum(num_lst, target, nums=2):
    num_to_idx = {}
    for i, num in enumerate(num_lst):
        complement = target - num
        if complement in num_to_idx:
            return [num_to_idx[complement], i]
        num_to_idx[num] = i


print(TwoSum([2,7,11,15], 9))
            
