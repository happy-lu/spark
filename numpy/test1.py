import numpy as np





if __name__ == '__main__':
    # a = np.random.rand(2, 3)
    # a = np.arange(15).reshape(3, 5)
    # print(a)
    # b = np.sum(a ** 2, axis=1)
    # c = b.reshape(1, 3)
    #
    #
    # aa=np.array([[1,0,1],[0,0,0]])
    # bb=[1,0,1]
    #
    #
    #
    # print(np.array_split(a,2, axis=1))

    import numpy as np

    a = np.arange(6)
    print(a)
    b = np.zeros((6, 2))
    print(b)
    b += np.reshape(a, (-1, 1))
    print(b)
    for i in range(2):
        # b[:,i] += a.T 不用转置也能正确广播
        b.T[i] += a
    print(b)

