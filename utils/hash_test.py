def my_hash(x):
    return sum(map(lambda a: ord(a)*37, list(x)))


def hash_str(str):
    hash_nums = []
    for each_str in str.split(" "):
        hash_nums.append(my_hash(each_str))
    print(sum(hash_nums))


if __name__ == '__main__':
    str1 = "select adc from table123 t1"
    hash_str(str1)

    str2 = "select abc from tadle123 t1"
    hash_str(str2)
