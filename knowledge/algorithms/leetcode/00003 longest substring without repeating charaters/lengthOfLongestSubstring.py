
"""
# 双指针算法模板
for (int i = 0, j = 0; i < n; ++i) {
    while (j < i && check(j, i)) {
        ++j;
    }
    // 具体问题的逻辑
}
"""


def lengthOfLongestSubstring(ins):
    ss = set()
    ans = i = 0
    for j,c in enumerate(ins):
        while c in ss:
            ss.remove(ins[i])
            i += 1
        ss.add(c)
        ans = max(ans, j-i+1)
    return ans
s = 'aabdabdkjbakbfiwvbfal'
z = lengthOfLongestSubstring(ins=s)
print(z)
