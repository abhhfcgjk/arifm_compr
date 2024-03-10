import numpy as np
# with open('t', 'ab') as f:
#     f.write(np.uint8(0).tobytes())
import os
# with open('t', 'rb') as sf, open('tmp', 'wb') as nf:
#     for byte in iter(lambda: sf.read(1), b''):
#         nf.write(byte)
#     nf.write(b'\x00')
# os.remove('tmp')

# with open('bit_compr.cmp','wb') as f:
#     byte = 0b01001100
#     print(byte.to_bytes(1, 'little'))
#     f.write(byte.to_bytes(1, 'little'))
import json

# with open('./bit.bnr', 'rb') as f, open('./fraq.json', 'r') as j:
#     lst: list[int] = []
#     # for byte in iter(lambda: f.read(1), b''):
#     #     if byte not in lst:
#     #             lst.append(byte)
#     # print(lst, len(lst))
#     data = json.load(j)
#     for elem in data:
#         print(hex(elem['byte']), elem['frequency'])
#     # print(data, len(data))

# e1, e2 = False, False

# with open('test_3', 'rb') as t1, open('bit2.bnr', 'rb') as t2, open('t', 'wb') as t:
#     c = 0
#     while(True):
#         if not e1:
#             elem1 = t1.read(1)
#         if not e2:
#             elem2 = t2.read(1)
#         e1 = elem1==b''
#         e2 = elem2==b''
#         if e1 and e2:
#             break
#         c += 1
#         t.write(elem1)
#         if elem1 != elem2:
#             print(c)
#             break
#     print(c)

# c = 0
# with open('test_3', 'rb') as f:
#     for elem in iter(lambda: f.read(1), b''):
#         print(elem, end='')
#         c += 1
#     print(c)
# with open('test_3', 'r') as f:
#     print(len(f.read()))
    # for i in range(2):
    # byte = f.read(1)
    # print(byte)
    # # byte = f.read(1)
    # # print(byte)
    # print(f.tell())
    # f.seek(0)
    # print(f.write(byte), byte)
    # print(f.tell())

# with open('t', 'wb') as f:
#     size = 272
#     f.write(size.to_bytes(3,'big'))
# with open('t', 'rb') as f:
#     re = f.read()
#     print(re[0:3])
#     print(re[1])
#     print(int.from_bytes(re[0:3],'big'))

# with open('bit.bnr', "wb")as f:
#     for i in range(100):
#         f.write(b'a')
#     for i in range(100):
#         f.write(b'b')
# import math
# print(math.sqrt(13))

# with open('test_3', 'rb') as f:
#     print(len(f.read()))


# from random import randint
# from time import time

# import numpy as np

# with open("t", "wb") as f:
#     t:np.int32 = np.int32(200001)
#     f.write(t.tobytes())
#     print(t)

# with open("t", "rb") as f:
#     # c = np.frombuffer(f.read(), dtype=np.uint8)
#     # c = np.array(f.read(), dtype=np.uint8)
#     c = np.fromfile(f, dtype=np.uint8)
#     print(c)
#     # print(c)
#     # t:np.int32 = np.frombuffer(c, dtype=np.uint32)[0]
#     # t1:np.uint8 = np.uint8(109)
#     # print(c, type(c))
    
#     # print(t1, type(t1))



# c = np.uint32(-1)
# print(bin(c >> 1))
# print(hex(c >> 1))
# print(10**0)

# import numpy as np

# len_context = 2
# # 10
# # 10*256 + 10
# # 10*256*256 + 10*256 + 2
# # print(np.cumprod(d[len_context:]*10), np.sum(np.cumsum(d[len_context:]*256)))
# d = np.array([1,1,10,10])
# summ = 0
# for c in d:
#     # summ *= 256
#     # summ += c
#     summ += 255*summ + c
# prev = np.zeros(4)
# summ = np.sum(np.cumsum(255*prev))
# print(summ)

# import numpy as np

# con = np.array([1,2,3,4,5,6,7])


# con = np.roll(con, -1)[-1] = 0

# con[-1] = 0
# print(con)

# class A:
#     def __init__(self) -> None:
#         self.a = 100
#     def d(self):
#         print("A",self.a)
#         self.a+=1

# class B:
#     def __init__(self, f:A) -> None:
#         self.e = 0
#         self.a =f
#     def p(self):
#         print("B", end=" ")
#         self.a.d()

# q = A()
# r = B(q)
# q.d()
# r.p()
# q.d()
# r.p()
# def f():
#     return False, 98

# _ = False
# if(l := f())[0]:
#     print(l[0], l[1])
# else:
#     print('booobs')

# n = np.uint32(75*256*256+79*256+86)
# a = (n & 0xffffff) >> 16
# b = (n & 0xffff) >> 8
# c = (n & 0xff) >> 0
# print(a.tobytes(), b.tobytes(), c.tobytes())

# cm_degree = 2
# i = cm_degree
# m = 256**(i+1) -1
# while(m):
#     print(np.uint8(((n & m)>> 8*i)).tobytes(), hex(m), i*8)
#     i -= 1
#     m >>= 8
# c = np.uint8(75)

# d = np.array([b'a', b'd', b'e'], dtype=bytes)
# d = np.append(d, c.tobytes())
# print(d)
# with open('t', 'wb') as f:
#     f.write(d)


def f(d:np.array):
    d[0] = 10

c = np.array([8],dtype=np.uint8)

f(c)
print(c[0])
