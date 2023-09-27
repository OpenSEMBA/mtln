class SFunction(object):
    def __init__(self, func):
            self.func = func
    def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
    def __add__(self, other):
            def summed(*args, **kwargs):
                    return self(*args, **kwargs) + other(*args, **kwargs)
            return summed

def add_t_functions(f1,f2):
    assert('t' in f1.__code__.co_varnames)
    assert('t' in f2.__code__.co_varnames)
    return lambda t : f1(t)+f2(t)
