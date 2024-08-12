# 示例：命名空间和作用域

# 全局变量
x = "global"

def outer_function():
    # 嵌套函数的外部函数变量
    x = "outer"

    def inner_function():
        # 嵌套函数变量
        nonlocal x  # 使用外部函数的变量
        x = "inner"
        print("Inner:", x)

    inner_function()
    print("Outer:", x)

outer_function()
print("Global:", x)

# Inner: inner
# Outer: inner
# Global: global
# x 在全局命名空间中是 "global"。

# outer_function 中的 x 在外部函数的命名空间中是 "outer"。
# inner_function 中的 x 使用 nonlocal 声明，
# 使其作用于外部函数的命名空间，所以它的值变为 "inner"。

