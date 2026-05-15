class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def AddTwoNums(l1, l2):
    dummy = ListNode(0)  # 虚拟头节点
    cur = dummy
    carry = 0

    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0  # 取 l1 的值
        val2 = l2.val if l2 else 0  # 取 l2 的值
        total = val1 + val2 + carry  # 计算总和

        carry = total // 10  # 计算进位
        cur.next = ListNode(total % 10)  # 创建新节点
        cur = cur.next  # 移动指针

        # 移动 l1 和 l2
        if l1: l1 = l1.next
        if l2: l2 = l2.next

    return dummy.next  # 返回结果链表

# 测试
def print_linked_list(node):
    while node:
        print(node.val, end=" -> ")
        node = node.next
    print("None")

l1 = ListNode(2, ListNode(4, ListNode(3)))  # 342
l2 = ListNode(5, ListNode(6, ListNode(4)))  # 465
result = AddTwoNums(l1, l2)  # 807
print_linked_list(result)  # 输出: 7 -> 0 -> 8 -> None