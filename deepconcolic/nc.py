from utils_io import *
from utils import *
from engine import (Input, TestTarget,
                    BoolMappedCoverableLayer, LayerLocalCriterion,
                    Criterion4RootedSearch,
                    Analyzer4RootedSearch)


# ---

# 表示神经元覆盖率的类
class NcLayer(BoolMappedCoverableLayer):
    '''
    Covered layer that tracks per-neuron activation.
    '''
    pass


# ---

# 表示神经元覆盖率target的类，这里会赋予默认值，不用关心
class NcTarget(NamedTuple, TestTarget):
    """Inherits :class:`engine.TestTarget` as well."""
    layer: NcLayer
    position: Tuple[int, ...]

    # 获得神经元的覆盖情况
    def cover(self, acts) -> None:
        self.layer.cover_neuron(self.position[1:])

    # 类的返回值
    def __repr__(self) -> str:
        return 'activation of {} in {}'.format(xtuple(self.position[1:]),
                                               self.layer)

    # 记录提示信息
    def log_repr(self) -> str:
        return '#layer: {} #pos: {}'.format(self.layer.layer_index,
                                            xtuple(self.position[1:]))

    # 评估输入
    def eval_inputs(self, inputs: Sequence[Input], eval_batch=None) \
            -> Sequence[float]:
        """
        Measures how a new input `t` improves towards fulfilling the
        target.  A negative returned value indicates that no progress is
        being achieved by the given input.
        """
        acts = eval_batch(inputs, layer_indexes=(self.layer.layer_index,))
        acts = acts[self.layer.layer_index][(Ellipsis,) + self.position[1:]]
        return acts

    # 评估输入是否有效
    def valid_inputs(self, evals: Sequence[float]) -> Sequence[bool]:
        return evals > 0


# ---

# 神经元覆盖率分析的类，表示当指标是神经元覆盖率的时候才有用那种约束求解方法
class NcAnalyzer(Analyzer4RootedSearch):
    '''
    Analyzer that finds inputs by focusing on a target within a
    designated layer.
    '''

    @abstractmethod
    def search_input_close_to(self, x: Input, target: NcTarget) -> Optional[Tuple[float, Input]]:
        raise NotImplementedError


# ---

# 神经元覆盖率标准的类，用于初始化神经元覆盖率指标
class NcCriterion(LayerLocalCriterion, Criterion4RootedSearch):
    """
    Neuron coverage criterion
    """

    # 构造函数
    def __init__(self, clayers: Sequence[NcLayer], analyzer: NcAnalyzer, **kwds):
        assert isinstance(analyzer, NcAnalyzer)
        super().__init__(clayers=clayers, analyzer=analyzer, **kwds)

    # 类的返回值
    def __repr__(self):
        return "NC"

    # 找到下一个满足测试target的具体测试用例
    def find_next_rooted_test_target(self) -> Tuple[Input, NcTarget]:
        cl, nc_pos, nc_value, test_case = self.get_max()
        cl.inhibit_activation(nc_pos)
        return test_case, NcTarget(cl, nc_pos[1:])


# ---

# 从engine里面导入setup, 以及Engine这个类，用于设置
from engine import setup as engine_setup, Engine

# 当覆盖率指标是神经元覆盖率的时候，用于设置参数
def setup(test_object=None,
          setup_analyzer: Callable[[dict], NcAnalyzer] = None,
          criterion_args: dict = {},
          **kwds) -> Engine:
    """
    Helper to build an engine for neuron-coverage (using
    :class:`NcCriterion` and an analyzer constructed using
    `setup_analyzer`).

    Extra arguments are passed to `setup_analyzer`.
    """

    # 把每一个层用对应的类设置好，降低耦合性
    setup_layer = (
        lambda l, i, **kwds: NcLayer(layer=l, layer_index=i,
                                     feature_indices=test_object.feature_indices,
                                     **kwds))
    # 获得覆盖的层s
    cover_layers = get_cover_layers(test_object.dnn, setup_layer,
                                    layer_indices=test_object.layer_indices,
                                    exclude_direct_input_succ=False)
    # 返回初始化好的参数设置
    return engine_setup(test_object=test_object,
                        cover_layers=cover_layers,
                        setup_analyzer=setup_analyzer,
                        setup_criterion=NcCriterion,
                        criterion_args=criterion_args,
                        **kwds)

# ---
