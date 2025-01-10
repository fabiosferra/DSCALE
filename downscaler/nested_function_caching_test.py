import inspect
from abc import ABC, abstractmethod
from functools import wraps

from joblib import Memory, memory

from downscaler import CONSTANTS

memory = Memory(CONSTANTS.CACHE_DIR)


class MemFunc:
    """Thin wrapper class around a function, to be used together with
    joblib.memory

    This wrapper is necessary because for cached functions that use other
    functions we need to keep track of the other functions, illustrated here:

    ```
    def f1(a, b):
    return a+b

    @memory.cache
    def f2(a, b):
        return f1(a, b)
    ```
    If `f1` changes, we have a problem since joblib.memory has no way of knowing
    that when looking at `f2`. The solution to the problem is giving `f1` as an
    input to `f2` **and** wrapping `f1` in this class.

    ```
    def f2(a, b, f1 : MemFunc = MemFunc(f1)):
        return f1(a, b)
    ```
    The only change needed is adding f1 as an input parameter of type MemFunc
    and setting the default to MemFunc(f1). As MemFunc implements a __call__
    method, the rest is completely user transparent.

    Internally, MemFunc saves the provided function, the source code of the
    function as well as its own source code. This last step is necessary to
    ensure that the wrapper MemFunc itself did not change.

    """

    def __init__(self, func):
        self.func = func
        self.func_code = inspect.getsource(func)
        self.Callable_code = inspect.getsource(MemFunc)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def make_cacheable(usecaching: bool = True):
    """This decorator makes caching using joblib.Memory.cache optional

    Parameters
    ----------
    usecaching : bool, optional
        Controls whether or not caching is used, by default True
    """

    def middle(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            return (memory.cache(fun) if usecaching else fun)(*args, **kwargs)

        return wrapper

    return middle


@make_cacheable(usecaching=True)
def f1(a, b):
    return a - b


class AbstractMemFunc(ABC):
    @staticmethod
    @abstractmethod
    def __call__():
        return

    def __getstate__(self) -> dict:

        state = self.__dict__.copy()
        state["function code"] = inspect.getsource(self.__class__)
        return state


class F1:
    def __new__(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    @staticmethod
    def __call__(a, b):
        return a * b


# ff1 = MemFunc(f1)


@make_cacheable
def f2(a, b, f1: MemFunc = MemFunc(f1)):
    print("Calling the actual function")
    return f1(a, b=b) + f1(a=a, b=b)


#
#
print(f2(2, 3))
print(f2(2, 3))
