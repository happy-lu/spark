import time


# func=before(hook_func)(func(args))
def before(hook_func):
    def _deco(func):
        def __deco(*args, **kwargs):
            print("before %s called." % func.__name__)
            hook_func(*args, **kwargs)
            return func(*args, **kwargs)

        return __deco

    return _deco


def after(hook_func):
    def _deco(func):
        def __deco(*args, **kwargs):
            result = func(*args, **kwargs)
            print("after %s called." % func.__name__)
            hook_func(*args, **kwargs)
            return result

        return __deco

    return _deco

class Test1():
    def call(self):
        print("test1.call")


t1=Test1()
def cal_time1(*arg1):
    def _deco(func):
        def __deco(*args, **kwargs):
            ctime = time.time()
            result = func(*args, **kwargs)
            print("use time:" + str(time.time() - ctime))
            arg1[0].call()
            return result

        return __deco

    return _deco


def cal_time2(**kwds):
    def decorate(f):
        for k in kwds:
            print(k)
            print(kwds[k])
        return f

    return decorate


def cal_time(func):
    def __deco(*args, **kwargs):
        ctime = time.time()
        result = func(*args, **kwargs)
        print("use time:" + str(time.time() - ctime))
        return result

    return __deco


def call():
    print("call")


@cal_time1(t1)
def test():
    for i in range(1):
        print("test")




if __name__ == '__main__':
    test()
