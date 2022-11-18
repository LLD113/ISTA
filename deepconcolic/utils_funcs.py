from typing import *
from collections import UserDict
import random
import numpy as np


# ---
# 返回元组长度
def xtuple(t):
    return t if len(t) > 1 else t[0]

# 将输入t列表化
def xlist(t):
    return [t] if t is not None else []

# 将t序列化成元组或列表，如果不是任何一种，序列化成列表返回
def seqx(t):
    return [] if t is None else t if isinstance(t, (list, tuple)) else [t]

# 返回x本身
def id(x):
    return x

# 如果a不是空返回a， 否则返回d；当第一个参数为空的情况下才返回第二个元素
def some(a, d):
    return a if a is not None else d

# f是一个函数，a不为空的时候返回f(a), 否则返回a本身也就是None
def appopt(f, a):
    return f(a) if a is not None else a

# ---


try:
    import pandas as pd
except:
    pd = None
    pass

# 将d从DataFrame格式变成numpy格式
def as_numpy(d):
    if pd is not None:
        # 调用isinstance(object, calssinfo)内置函数判断是不是一个pandas类型的数据，也就是DataFrame, 如果是，转换成numpy类型的数据返回
        # 否则返回本身
        return d.to_numpy() if isinstance(d, pd.core.frame.DataFrame) else d
    else:
        return d


# ---

# 设置随机种子
def rng_seed(seed: Optional[int]):
    if seed is None:
        seed = int(np.random.uniform(2 ** 32 - 1))
    print('RNG seed:', seed)  # Log seed to help some level of reproducibility
    np.random.seed(seed)
    # In case one also uses pythons' stdlib ?
    random.seed(seed)

# 产生随机的整形
def randint():
    return int(np.random.uniform(2 ** 32 - 1))


# ---


try:
    # Use xxhash if available as it's probably more efficient
    import xxhash

    __h = xxhash.xxh64()

    # 将numpy哈希化
    def np_hash(x):
        __h.reset()
        __h.update(x)
        return __h.digest()
except:
    def np_hash(x):
        return hash(x.tobytes())
    # NB: In case we experience too many collisions:
    # import hashlib
    # def np_hash (x):
    #   return hashlib.md5 (x).digest ()

# 自定义一个支持numpy array作为键值的字典
class NPArrayDict(UserDict):
    '''
    Custom dictionary that accepts numpy arrays as keys.
    '''

    def __getitem__(self, x: np.ndarray):
        return self.data[np_hash(x)]

    def __delitem__(self, x: np.ndarray):
        del self.data[np_hash(x)]

    def __setitem__(self, x: np.ndarray, val):
        x.flags.writeable = False
        self.data[np_hash(x)] = val

    def __contains__(self, x: np.ndarray):
        return np_hash(x) in self.data


# ---


D, C = TypeVar('D'), TypeVar('C')

# 未知域上的延迟评估
class LazyLambda:
    '''
    Lazy eval on an unknown domain.
    '''

    def __init__(self, f: Callable[[D], C], **kwds):
        super().__init__(**kwds)
        self.f = f

    def __getitem__(self, x: D) -> C:
        return self.f(x)

    def __len__(self) -> int:
        return self.f(None)

# 固定域上的惰性函数求值。
class LazyLambdaDict(Dict[D, C]):
    '''
    Lazy function eval on a fixed domain.
    '''

    def __init__(self, f: Callable[[D], C], domain: Set[D], **kwds) -> Dict[D, C]:
        super().__init__(**kwds)
        self.domain = domain
        self.f = f

    def __getitem__(self, x: D) -> D:
        if x not in self.domain:
            return KeyError
        return self.f(x)

    def __contains__(self, x: D) -> bool:
        return x in self.domain

    def __iter__(self) -> Iterator[D]:
        return self.domain.__iter__()

    def __setitem__(self, _):
        raise RuntimeError('Invalid item assignment on `LazyLambdaDict` object')

    def __delitem__(self, _):
        raise RuntimeError('Invalid item deletion on `LazyLambdaDict` object')
