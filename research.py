import glm
import timeit

def regular():
    x = [1, 2, 3]
    y = (*x, )

def custom():
    x = [1, 2, 3]
    y = x[0], x[1], x[2]

reg = timeit.timeit(regular)
custom = timeit.timeit(custom)

print(reg)
print(custom)
