from utils_io import *
from utils_funcs import *
from utils import *
from functools import reduce
from sklearn.model_selection import train_test_split
import yaml  # for dumping record
import hashlib  # for hashing test inputs

# ---


# Define an alias type for inputs
# 定义类型别名
Input = NewType("Input", np.ndarray)

# 输入字典，空类
class InputsDict(NPArrayDict):
    pass


# ---

# 私有类，输入状态初始状态，为实现多态的基类，不用关心
class _InputsStatBasedInitializable:

    @abstractmethod
    def inputs_stat_initialize(self,
                               train_data: raw_datat = None,
                               test_data: raw_datat = None) -> None:
        print(self)
        raise NotImplementedError


# ---
# 私有类，激活状态初始化，用于实现多态，不用太关心
class _ActivationStatBasedInitializable:

    def stat_based_basic_initializers(self):
        """
        Stat-based initialization steps (non-batched).

        函数返回一个字典包括：
            名字：
            层序：
            只调用一次的map
            打印（可选的）：打印一个概述结果
        Returns a list of dictionaries (or `None`) with the following
        entries:

        - name: short description of what's computed;

        - layer_indexes: a list or set of indexes for layers whose
          activations values are needed;

        - once: a callable taking a mapping (as a dictionary) from each
          layer index given in `layer_indexes` to activation values for
          the corresponding layer; this is to be called only once during
          initialization of the analyzer;

        - print (optional): a function that prints a summary of results.
        """
        return []

    # 返回一个列表：列表的内容是字典，用于增量初始化的
    def stat_based_incremental_initializers(self):
        """
        Stat-based incremental initialization steps.

        Returns a list of dictionaries (or `None`) with the following
        entries:

        - name: short description of what's computed;

        - layer_indexes: a list or set of indexes for layers whose
          activations values are needed;

        - accum: a callable taking batched activation values for every
          layer and any accumulator that is (initially `None`), and
          returns a new or updated accumulator.  This is called at least
          once.

        - final: optional function that is called with the final
          accumulator once all batched activations have been passed to
          `accum`;

        - print (optional): a function that prints a summary of results.
        """
        return []

    # 用于训练数据初始化
    def stat_based_train_cv_initializers(self):
        """
        Stat-based initialization steps with optional cross-validation
        performed on training data.

        Returns a list of dictionaries (or `None`) with the following
        entries:

        - name: short description of what's computed;

        - layer_indexes: a list or set of indexes for layers whose
          activations values are needed;

        - test_size & train_size: (as in
          `sklearn.model_selection.train_test_split`)

        - train: a callable taking some training data as mapping (as a
          dictionary) from each layer index given in `layer_indexes` to
          activation values for the corresponding layer, and two keyword
          arguments `true_labels` and `pred_labels` that hold the
          corresponding true and predicted labels. Returns some arbitrary
          object, and is to be called only once during initialization;

        - test: a callable taking some extra training data and associated
          labels as two separate mappings (as a dictionary) from each
          layer index given in `layer_indexes` to activation values for
          the corresponding layer.

        Any function given as entries above is always called before
        functions returned by `stat_based_test_cv_initializers`, but after
        those retured by `stat_based_basic_initializers` and
        `stat_based_incremental_initializers`.
        """
        return []

    # 测试数据初始化
    def stat_based_test_cv_initializers(self):
        """
        Stat-based initialization steps with optional cross-validation
        performed on test data.

        Returns a list of dictionaries (or `None`) with the following
        entries:

        - name: short description of what's computed;

        - layer_indexes: a list or set of indexes for layers whose
          activations values are needed;

        - test_size & train_size: (as in
          `sklearn.model_selection.train_test_split`)

        - train: a callable taking some test data as mapping (as a
          dictionary) from each layer index given in `layer_indexes` to
          activation values for the corresponding layer, and two keyword
          arguments `true_labels` and `pred_labels` that hold the
          corresponding true and predicted labels. Returns some arbitrary
          object, and is to be called only once during initialization;

        - test: a callable taking some extra test data and associated
          labels as two separate mappings (as a dictionary) from each
          layer index given in `layer_indexes` to activation values for
          the corresponding layer.

        Any function given as entries above is always called last.
        """
        # - accum_test: a callable that is called with the object returned
        #   by `train`, along with batched activation values for every layer
        #   on the test data, and returns a new or updated accumulator.
        #   This is called at least once.
        #
        # - final_test: optional function that is called with the final test
        #   accumulator once all batched test activations have been passed
        #   to  `accum_test`.
        return []


# ---

# 静态过滤器，用于扩展
class StaticFilter:
    '''
    A static filter can be used to compare any concrete input against a
    pre-computed dataset.
    '''

    @abstractmethod
    def close_enough(self, x: Input) -> bool:
        raise NotImplementedError


# ---

# 动态过滤器，抽象类用于实现多态，用于扩展
class DynamicFilter:
    '''
    A dynamic filter can be used to compare any concrete input against a
    given reference set.
    '''

    @abstractmethod
    def close_to(self, refs: Sequence[Input], x: Input) -> bool:
        raise NotImplementedError


# ---

# 用于表示边界的类，任何需要边界表示的对象可以用这个类来实现
class Bounds:
    """
    Basic abstract class to represent any bounds.  (Mostly for typing
    arguments and sub-classing.)
    """

    @property
    def low(self) -> np.array(float):
        raise NotImplementedError

    @property
    def up(self) -> np.array(float):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, _idx: Tuple[int, ...]) -> Tuple[float, float]:
        raise NotImplementedError


# ---

# 用于表示覆盖率的类
class Coverage:
    """Basic helper class to manipulate and type-annotate coverage measures."""

    # 构造函数
    def __init__(self, covered=None, total=None, non_covered=None):
        if total != None:
            self.total = total
        elif covered != None and non_covered != None:
            self.total = covered + non_covered
        elif covered != None:
            self.total = covered
        elif non_covered != None:
            self.total = non_covered
        else:
            self.total = 0

        if covered != None:
            self.c = covered
        elif non_covered != None and self.total > 0:
            self.c = self.total - non_covered
        else:
            self.c = 0

    # 增加项
    def __add__(self, x):
        return Coverage(covered=self.c + x.c,
                        total=self.total + x.total)

    # 乘一个数字
    def __mul__(self, f: float):
        return Coverage(covered=float(self.c) * f,
                        total=self.total)

    # 返回是否计算完成，及测试用例c是否等于全部的测试用例
    @property
    def done(self) -> bool:
        return self.total == self.c

    # 返回覆盖率的计算
    @property
    def as_prop(self) -> float:
        return (((1.0 * self.c) / (1.0 * self.total))
                if self.total != 0 else 0.0)

    # 类的返回值，返回覆盖率的计算
    def __repr__(self):
        return str(self.as_prop)


# ---

# 用于拓展的类，暂时等价Criterion
class Metric(DynamicFilter):
    '''
    For now, we can assume that every metric can also be used as a
    filter to compare and assess concrete inputs.
    '''

    # 构造函数
    def __init__(self, factor=0.25, scale=1, **kwds):
        '''
        The `factor` argument determines closeness when the object is used
        as a filter; defaults to 1/4.  In turn, `scale` is applied on
        involved scalar values (e.g. pixels) when computing distances.
        '''
        self.factor = factor
        self.scale = scale
        super().__init__(**kwds)

    # 抽象方法，用于计算距离
    @abstractmethod
    def distance(self, x, y):
        '''
        Returns the distance between two concrete inputs `x` and `y`.
        '''
        raise NotImplementedError

    # 属性方法，判断是不是int
    @property
    def is_int(self):
        '''
        Holds iff Integer metrics.
        '''
        return False


# ---

# 测试目标类，用于记录
class TestTarget:
    '''
    Base record of test targets.
    '''

    # 抽象方法，覆盖
    @abstractmethod
    def cover(self, acts) -> None:
        '''
        Record that the target has been covered by the given set of
        activations.
        '''
        raise NotImplementedError

    # 日志方法，抽象方法
    def log_repr(self) -> str:
        '''
        Returns a single-line string representation of the target suitable
        for logging.
        '''
        raise NotImplementedError


# ---

# 约束求解器总类，相当于最基础的类，所有的类都要继承它
class Analyzer:
    '''
    Base class for any kind of analyzer that is able to construct new
    concrete inputs.
    '''

    # 构造函数
    def __init__(self,
                 analyzed_dnn=None,
                 input_bounds: Optional[Bounds] = None,
                 postproc_inputs: Callable[[Sequence[Input]], Sequence[Input]] = id,
                 **kwds):
        assert analyzed_dnn is not None
        assert input_bounds is None or isinstance(input_bounds, Bounds)
        self._analyzed_dnn = analyzed_dnn
        self._input_bounds = input_bounds
        self._postproc_inputs = postproc_inputs
        super().__init__(**kwds)

    # ---

    # TODO: `dnn` and the two methods below (previously in test_objectt)
    # would deserve to be on their own as they are not strictly speaking
    # analyzer-dependent.  Yet they stay there for now as analyzers,
    # criteria, and engines rely on at least one of them.

    # 属性方法，返回一个分析后的dnn模型
    @property
    def dnn(self) -> keras.Model:
        '''
        The analyzed DNN.
        '''
        return self._analyzed_dnn

    # evaluation评估函数， 返回对于给定的input激活函数情况
    def eval(self, i, **kwds):
        '''
        Returns the activations associated to a given input.
        '''
        return eval(self.dnn, i, **kwds) # 这个eval是utils里面的

    # 评估批次，返回对于一个给定的输入批次的激活函数情况
    def eval_batch(self, i, **kwds):
        '''
        Returns all activations associated to a given input batch.
        '''
        return eval_batch(self.dnn, i, **kwds)

    # 抽象方法，返用于比较具体输入的度量
    @abstractmethod
    def input_metric(self) -> Metric:
        '''
        Returns the metric used to compare concrete inputs.
        '''
        raise NotImplementedError

    # 属性方法：返回边界，扩展方法
    @property
    def input_bounds(self) -> List[Bounds]:
        '''
        Returns the bounds on generated inputs.
        '''
        return [self._input_bounds] if self._input_bounds is not None else []


# ---

# 约束求解器之一，与Criterion4RootedSearch含义类似
class Analyzer4RootedSearch(Analyzer):
    '''
    Analyzers that are able to find new concrete inputs close to a given
    input should inherit this class.
    '''

    # 抽象方法：返回要给元组（d, y）, y 是一个具体的用例，d是给定度量方式下的度量距离
    # 子类需要继承并实现这个方法
    @abstractmethod
    def search_input_close_to(self, x, target: TestTarget) -> Optional[Tuple[float, Input]]:
        '''
        Generates a new concrete input close to `x`, that fulfills test
        target `target`.

        Returns a tuple `(d, y)`, that is a new concrete input `y` along
        with its distance `d` w.r.t the input metric, or `None` is
        unsuccessful.
        '''
        raise NotImplementedError


# ---

# 约束求解器之一，与Criterion4FreeSearch类似
class Analyzer4FreeSearch(Analyzer):
    '''
    Analyzers that are able to find new concrete inputs close to any
    input from a give set of test cases.
    '''

    @abstractmethod
    def search_close_inputs(self, target: TestTarget) -> Optional[Tuple[float, Input, Input]]:
        '''
        Generates a new concrete input that fulfills test target `target`.

        Returns a tuple `(d, base, new)` where `base` is a concrete
        element from a set given on initialization (typically for now, raw
        data from `test_object`) and `new` is a new concrete input at
        distance `d` from `base`, or `None` is unsuccessful.
        '''
        raise NotImplementedError


# ---

# 报告文件对象写入前都用这个类表示
class Report:
    '''
    A simple class to take reporting stuff out from the engine.
    '''

    def __init__(self,
                 base_name='',
                 outdir: OutputDir = None,
                 save_new_tests=False,
                 adv_dist_period=100,
                 save_input_func=None,
                 amplify_diffs=False,
                 inp_up=1,  # XXX: ??? unused.
                 **kwds):

        self.adversarials = []
        self.base_name = base_name
        self.save_new_tests = save_new_tests
        self.adv_dist_period = adv_dist_period
        self.outdir = outdir or OutputDir()
        assert isinstance(self.outdir, OutputDir)
        self.base = self.outdir.stamped_filename(self.base_name)
        self.report_file = self.outdir.filepath(self.base + '_report.txt')
        self.save_input_func = save_input_func
        self.amplify_diffs = amplify_diffs
        p1('Reporting into: {0}'.format(self.report_file))
        self.ntests = 0
        self.nsteps = 0

    # 私有方法：保存输入
    def _save_input(self, im, name, log=None):
        if self.save_input_func != None:
            self.save_input_func(im, name, self.outdir.path, log)

    # 私有方法：保存衍生的输入
    def _save_derived_input(self, new, origin, diff=None, log=None):
        self._save_input(new[0], new[1], log)
        self._save_input(origin[0], origin[1], log)
        if diff is not None:
            self._save_input(diff[0], diff[1], log)

    # 公开的save_input方法
    def save_input(self, i, suff):
        self._save_input(i, self.base + '_' + suff)

    # 新的test, 记录产生的新测试用例
    def new_test(self, new=(), orig=(), dist=None, is_int=None):
        if self.save_new_tests:
            if self.amplify_diffs:
                diff = np.abs(new[0] - orig[0])
                diff *= 0.5 / np.max(diff)
            else:
                diff = new[0] - orig[0]
            self._save_derived_input((new[0], '{0.ntests}-ok-{1}'.format(self, new[1])),
                                     (orig[0], '{0.ntests}-original-{1}'.format(self, orig[1])),
                                     (diff, '{0.ntests}-diff-{1}'.format(self, orig[1])))
        self.ntests += 1

    # 属性方法，记录到第几步了
    @property
    def num_steps(self):
        return self.nsteps

    # 属性方法：记录到第几个测试用例了
    @property
    def num_tests(self):
        return self.ntests

    # 属性方法：记录是对抗样本的数量
    @property
    def num_adversarials(self):
        return len(self.adversarials)

    def new_adversarial(self, new=(), orig=(), dist=None, is_int=None):
        self.adversarials.append((orig, new, dist))
        self._save_derived_input((new[0], '{0.ntests}-adv-{1}'.format(self, new[1])),
                                 (orig[0], '{0.ntests}-original-{1}'.format(self, orig[1])),
                                 (np.abs(new[0] - orig[0]), '{0.ntests}-diff-{1}'.format(self, orig[1])))
        if self.num_adversarials % self.adv_dist_period == 0:
            print_adversarial_distribution(
                [d for o, n, d in self.adversarials],
                self.outdir.filepath(self.base + '_adversarial-distribution.txt'),
                int_flag=is_int)
        self.ntests += 1

    # 打印一个报告行，不要进行到下一步，除非dry参数为true
    def step(self, *args, dry=False) -> None:
        '''
        Prints a single report line.

        Do not count as new step if `dry` holds.
        '''
        append_in_file(self.report_file, *args, '\n')
        if not dry:
            self.nsteps += 1

    # 记录方法：输出一个关于多有初始的以及生成的测试用例的记录
    def record(self, test_cases, record, **kwds) -> None:
        """
        Outputs a record about all initial and generated test cases.

        The record essentually encodes the tree that enables one to trace
        the origins of all generated tests.
        """
        tests = [dict(**record[x],
                      md5=hashlib.md5(x).hexdigest())
                 for x in test_cases]
        advrs = [dict(**record[n[0]],
                      md5=hashlib.md5(n[0]).hexdigest())
                 for _o, n, _d in self.adversarials]
        data = dict(passed_tests=tests,
                    adversarials=advrs,
                    **kwds)
        path = self.outdir.stamped_filepath('record', suff='.yml')
        with open(path, 'w') as f:
            yaml.dump(data, f)


# ---

# 拓展类，用于实现提前终止
class EarlyTermination(Exception):
    '''
    Exception raised by criteria when no new test target can be found.
    '''
    pass


# ---

# 覆盖的层相关的对象用这个类表示
class CoverableLayer:
    '''
    Base class for any layer based on which coverability criteria are
    defined.
    '''

    # 构造函数
    def __init__(self, layer=None, layer_index=None,
                 prev: int = None, succ: int = None):
        self.layer = layer
        self.layer_index = layer_index
        self.is_conv = is_conv_layer(layer)
        self.prev_layer_index = prev
        self.succ_layer_index = succ

    # 获取层的信息
    def get_info(self):
        return dict(layer_name=self.layer.name,
                    layer_index=self.layer_index,
                    prev_layer_index=self.prev_layer_index,
                    succ_layer_index=self.succ_layer_index)

    # 设置层的信息
    def set_info(self, dnn,
                 layer_name=None,
                 layer_index=None,
                 prev_layer_index=None,
                 succ_layer_index=None):
        self.layer = dnn.get_layer(name=layer_name)
        self.layer_index = layer_index
        self.is_conv = is_conv_layer(self.layer)
        self.prev_layer_index = prev_layer_index
        self.succ_layer_index = succ_layer_index

    # 类的返回值，返回层的名字
    def __repr__(self):
        return self.layer.name


# ---

# 记录目标的水平为1
_log_target_selection_level = 1

# 覆盖准则相关的对象用这个类表示
class Criterion(_ActivationStatBasedInitializable):
    '''
    Base class for test critieria.

    Note that a criterion MUST inherit either (or both)
    :class:`Criterion4FreeSearch` or :class:`Criterion4RootedSearch`.
    '''

    # 构造函数
    def __init__(self,
                 clayers: Sequence[CoverableLayer],
                 *args,
                 analyzer: Analyzer = None,
                 prefer_rooted_search: bool = None,
                 verbose: int = 1,
                 **kwds):
        '''
        A criterion operates based on an `analyzer` to find new concrete
        inputs.

        Flag `prefer_rooted_search` can be used in case both the criterion
        and the analyzer support the two kinds of search; the default
        behavior is to select rooted search.
        '''
        assert isinstance(analyzer, Analyzer)
        super().__init__(*args, **kwds)
        self.cover_layers = clayers
        self.analyzer = analyzer
        self.test_cases = []
        self.verbose = some(verbose, 1)
        self.rooted_search = self._rooted_search(prefer_rooted_search)

    # True for rooted search, False for free search
    # 判断用那个搜索方式，这里用rooted search
    def _rooted_search(self, prefer_rooted_search=None):
        '''
        Holds if rooted-search mode is selected and the criterion and
        analyzer pair supports it.

        Parameters
        ----------
        prefer_rooted_search: bool, optional

        Returns
        -------
        whether rooted search mode is selected.

        '''
        rooted_ok = (isinstance(self.analyzer, Analyzer4RootedSearch) and
                     isinstance(self, Criterion4RootedSearch))
        free_ok = (isinstance(self.analyzer, Analyzer4FreeSearch) and
                   isinstance(self, Criterion4FreeSearch))
        if not (free_ok or rooted_ok):
            sys.exit('Incompatible pair criterion/analyzer')
        if free_ok and rooted_ok and prefer_rooted_search is None:
            p1('Arbitrarily selecting rooted search against free search.')
        return rooted_ok and (prefer_rooted_search is None or prefer_rooted_search)

    # ---

    # 抽象方法，继承的类必须实现这个方法
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    # 最终设置
    @abstractmethod
    def finalize_setup(self) -> None:
        """
        Called once after any stat-based initialization (see, e.g.,
        :meth:`stat_based_basic_initializers`), and before any call to
        :meth:`add_new_test_cases`, :meth:`coverage`, and
        :meth:`search_next`.
        """
        raise NotImplementedError
        # self.analyzer.finalize_setup ()

    # 终止进程的函数
    def terminate(self) -> None:
        """
        Called once upon termination of the test case generation.

        Use this to dump some files or reports.
        """
        pass

    # 重置测试用例集合
    def reset(self):
        '''
        Empties the set of test cases
        '''
        self.test_cases = []

    # 抽象方法，返回覆盖率，继承类必须要实现这个方法
    @abstractmethod
    def coverage(self) -> Coverage:
        '''
        Returns a measure of the current coverage.
        '''
        raise NotImplementedError

    # 属性方法返回该分析器所用的度量标准
    @property
    def metric(self) -> Metric:
        '''
        Returns the metric used by the analyzer to compare concrete
        inputs.
        '''
        return self.analyzer.input_metric()

    # 测试用例的个数
    @property
    def num_test_cases(self) -> int:
        '''
        Returns the number of test cases.
        '''
        return len(self.test_cases)

    # 弹出测试用例
    def pop_test(self) -> None:
        '''
        Removes the last registered test (while keeping its coverage).
        '''
        self.test_cases.pop()

    # 抽象方法，返回覆盖的层的索引，继承类必须都要实现这个方法
    @abstractmethod
    def covered_layer_indexes(self) -> List[int]:
        return [cl.layer_index for cl in self.cover_layers]

    # final as well 增加新的测试用例
    def add_new_test_cases(self, tl: Sequence[Input],
                           covered_target: TestTarget = None) -> None:
        """
        As its name says, this method adds a given series of inputs into
        the set of test cases.  It then calls :meth:`register_new_activations`.
        """
        tp1('Adding {} test case{}'.format(*s_(len(tl))))
        self.test_cases.extend(tl)
        layer_indexes = self.covered_layer_indexes()
        for acts in self._batched_activations(tl, allow_input_layer=False,
                                              layer_indexes=layer_indexes):
            if covered_target is not None:
                covered_target.cover(acts)
            self.register_new_activations(acts)

    # 私有方法，每批次激活函数的情况
    def _batched_activations(self, tl: Sequence[Input], **kwds) -> range:
        batches = np.array_split(tl, len(tl) // 100 + 1)
        for batch in batches:
            acts = self.analyzer.eval_batch(batch, **kwds)
            yield (acts)
            del acts

    # 抽象方法，新的测试用例产生的时候调用这个方法更新覆盖率，继承的类要实现这个抽象方法
    @abstractmethod
    def register_new_activations(self, acts) -> None:
        """
        Method called whenever new test cases are registered.  Overload
        this method to update coverage.
        """
        raise NotImplementedError

    # 搜索下一个满足度量要求的测试用例
    def search_next(self) -> Tuple[Union[Tuple[Input, Input, float], None], TestTarget]:
        '''
        Selects a new test target based (see
        :class:`Criterion4RootedSearch` and
        :class:`Criterion4FreeSearch`), and then uses the analyzer to find
        a new concrete input.

        Returns a pair of:

        - either `None` in case of failure of the analyzer, or a triple
          `(x0, x1, d)`, `x1` being the new concrete input generated by
          the analyzer;

        - the test target considered.
        '''
        if self.rooted_search:
            x0, target = self.find_next_rooted_test_target()
            if self.verbose >= _log_target_selection_level:
                p1(f'| Targeting {target}')
            x1_attempt = self.analyzer.search_input_close_to(x0, target)
            if x1_attempt == None:
                return None, target
            else:
                d, x1 = x1_attempt
                return (x0, x1, d), target
        else:
            target = self.find_next_test_target()
            if self.verbose >= _log_target_selection_level:
                p1(f'| Targeting {target}')
            attempt = self.analyzer.search_close_inputs(target)
            if attempt == None:
                return None, target
            else:
                d, x0, x1 = attempt
                return (x0, x1, d), target

    # ---


# ---

# 与之前层有关联的，并且有覆盖准则目标值的，用这个类来实现，也是用于实现多态以及扩展性定义的类
class Criterion4RootedSearch(Criterion):
    '''
    Any criterion that can be used to find a pair of a base test case
    and a test target should inherit this class.
    '''

    # 抽象方法，从记录的测试用例集中寻找与现有测试输入相关联的新测试目标。
    @abstractmethod
    def find_next_rooted_test_target(self) -> Tuple[Input, TestTarget]:
        '''
        Seeks a new test target associated with an existing test input
        taken from the set of recorded test cases.

        Note this method MUST perform enough bookkeeping so that two
        successive calls that are not interleaved with any call to
        :meth:`Criterion.add_new_test_cases` return different results.
        This property is to enforce progress upon unsuccessful search of
        concrete inputs.
        '''
        raise NotImplementedError


# ---

# 与之前层无关的覆盖率用这个计算，也是用于实现多态以及扩展性定义的类
class Criterion4FreeSearch(Criterion):
    '''
    Any criterion that can be used to select a test target without
    relying on activation data or previously inserted test cases should
    inherit this class.
    '''

    @abstractmethod
    def find_next_test_target(self) -> TestTarget:
        '''
        Seeks and returns a new test target.
        '''
        raise NotImplementedError


# ---

# 用给定的覆盖准则返回一个基础报告形式
def setup_basic_report(criterion: Criterion, **kwds) -> Report:
    '''
    Returns a very basic report object that feeds files whose base names
    are constructed from the provided criterion.

    Extra keyword arguments are passed on to the constructor of
    :class:`Report`.
    '''
    return Report(base_name='{0}_{0.metric}'.format(criterion), **kwds)


# ---

# 引擎类，整个deepconcolic的核心类
class Engine:
    '''
    Core Deepconcolic engine.
    '''

    # 构造函数初始化，包括初始化测试数据，训练数据，衡量标准，自定义的过滤器等
    def __init__(self, test_data, train_data,
                 criterion: Criterion,
                 custom_filters: Sequence[Union[StaticFilter, DynamicFilter]] = [],
                 **kwds):
        """
        Builds a test engine with the given DNN, reference data, and test
        criterion.  Uses the input metric provided by the
        criterion-specific analyzer as filter for assessing legitimacy of
        new test inputs, unless `custom_filters` is not `None`.
        """
        self.ref_data = test_data
        self.train_data = train_data
        self.criterion = criterion
        fltrs = [criterion.metric]
        fltrs += custom_filters \
            if isinstance(custom_filters, list) \
            else [custom_filters]
        # NB: note some filters may belong to both lists:
        self.static_filters = [f for f in fltrs if isinstance(f, StaticFilter)]
        self.dynamic_filters = [f for f in fltrs if isinstance(f, DynamicFilter)]
        super().__init__(**kwds)
        # 状态初始化
        self._stat_based_inits()
        self._initialized = False

    # 类的返回值，返回提示信息
    def __repr__(self):
        return 'criterion {0} with norm {0.metric}'.format(self.criterion)

    # 私有方法：返回单个测试用例预测结果
    def _run_test(self, x):
        return prediction(self.criterion.analyzer.dnn, x)

    # 私有方法，返回多个测试用例预测结果
    def _run_tests(self, xl):
        return predictions(self.criterion.analyzer.dnn, xl)

    # 私有方法：搜索钱需要调用一次，初始化所有的类型
    def _initialize_search(self, report: Report, initial_test_cases=None):
        '''
        Method called once at the beginning of search.
        '''
        xl = []
        if initial_test_cases is not None and initial_test_cases > 0:
            x = self.ref_data.data
            if self.ref_data.labels is not None:
                x = x[self._run_tests(x) == self.ref_data.labels]
            x = np.random.default_rng().choice(a=x, axis=0,
                                               size=min(initial_test_cases, len(x)))
            if self.ref_data.labels is not None:
                p1('Initializing with {} randomly selected test case{} that {} correctly classified.'
                   .format(*s_(len(x)), is_are_(len(x))[1]))
            else:
                p1('Initializing with {} randomly selected test case{}.'
                   .format(*s_(len(x))))
            self.criterion.add_new_test_cases(x)
        elif initial_test_cases is None and self.criterion.rooted_search:
            p1('Randomly selecting an input from test data.')
            x = np.random.default_rng().choice(a=self.ref_data.data, axis=0)
            report.save_input(x, 'seed-input')
            self.criterion.add_new_test_cases([x])

    # 运行函数，回初始化数据，调用约束求解以及覆盖率相关函数
    def run(self,
            report: Union[Report, Callable[[Criterion], Report]] = setup_basic_report,
            initial_test_cases=None,
            check_root_only: bool = True,
            max_iterations=-1,
            **kwds) -> Report:
        '''
        Uses `report` to construct a helper for outputing logs and new
        test cases, and then starts the engine for either: up to
        `max_iterations` iterations (i.e. number of runs of the analyzer)
        if `max_iterations >= 0`, or else until full coverage is reached,
        or the criterion is fulfilled (whichever happens first).

        Set `check_root_only` to `False` to ensure every new generated
        test case that is close enough to any reference test data is
        kept. Leaving it to `True` speeds the oracle check by only
        comparing new tests agains the original reference version.
        '''

        # 表示指定的覆盖率
        criterion = self.criterion

        # 如果没有初始化，初始化
        if not self._initialized:
            criterion.finalize_setup()
            # 打印提示信息，以几个测试用例开始，运行几轮
            p1('Starting tests for {}{}.'
               .format(self, '' if max_iterations < 0 else
            ' ({} max iterations)'.format(max_iterations)))
            self._initialized = True
        # 初始化完成，直接继续运行
        else:
            p1('Continuing tests for {}{}.'
               .format(self, '' if max_iterations < 0 else
            ' ({} max iterations)'.format(max_iterations)))
            initial_test_cases = initial_test_cases or 0

        # 生成报告文件
        report = report if isinstance(report, Report) else \
            report(criterion, **kwds)

        # Initialize search to add new test cases in every call to run:
        self._initialize_search(report, initial_test_cases)

        # 计算覆盖率
        coverage = criterion.coverage()
        p1('#0 {}: {.as_prop:10.8%}'.format(criterion, coverage))
        report.step('{0}-cover: {1} #test cases: {0.num_test_cases} '
                    .format(criterion, coverage),
                    '#adversarial examples: 0',
                    dry=True)

        iteration = 1
        init_tests = report.num_tests
        init_adversarials = report.num_adversarials
        # Note some test cases might be inserted multiple times: in such a
        # case only the max index will be remembered as origin:
        record = InputsDict([(x, dict(root_index=i,
                                      index=i,
                                      label=int(self._run_test(x))))
                             for i, x in enumerate(criterion.test_cases)]) \
            if criterion.rooted_search else InputsDict()
        check_root_only &= criterion.rooted_search

        try:
            # 循环调用, 当迭代次数没有超出，且覆盖率计算没有完成，都要继续循环
            while ((iteration <= max_iterations or max_iterations < 0) and
                   not coverage.done):

                adversarial = False

                search_attempt, target = criterion.search_next()
                if search_attempt != None:
                    x0, x1, d = search_attempt

                    # Check if x1 is already met:
                    new = x1 not in record

                    # Test oracle for adversarial testing
                    close_enough = new
                    close_enough &= all(f.close_to(self.ref_data.data if not check_root_only else
                                                   [criterion.test_cases[record[x0]['root_index']]], x1)
                                        for f in self.dynamic_filters)
                    close_enough &= all(f.close_enough(x1) for f in self.static_filters)
                    if close_enough:
                        criterion.add_new_test_cases([x1], covered_target=target)
                        coverage = criterion.coverage()
                        y0 = self._run_test(x0)
                        y1 = self._run_test(x1)
                        if criterion.rooted_search:
                            root_index = record[x0]['root_index']
                            root_dist = criterion.metric.distance(x1, criterion.test_cases[root_index])
                            record[x1] = dict(root_index=root_index,
                                              root_dist=float(root_dist),
                                              origin_index=record[x0]['index'],
                                              origin_dist=float(d),
                                              gen_test_id=report.num_tests,
                                              index=len(record),
                                              label=int(y1))

                        if y1 != y0:
                            adversarial = True
                            criterion.pop_test()
                            report.new_adversarial(new=(x1, y1), orig=(x0, y0), dist=d,
                                                   is_int=criterion.metric.is_int)
                        else:
                            report.new_test(new=(x1, y1), orig=(x0, y0), dist=d,
                                            is_int=criterion.metric.is_int)

                # print("I am a test for NC:100")
                # 打印提示信息
                p1('#{} {}: {.as_prop:10.8%} {}'
                   .format(iteration, criterion, coverage,
                           'with {} at {} distance {}: {}'
                           .format('new test case' if close_enough else
                                   'failed attempt' if new else 'not new',
                                   criterion.metric, d,
                                   'too far from raw input' if (not close_enough and
                                                                not check_root_only) else
                                   'too far from original input' if not close_enough else
                                   'adversarial' if adversarial else 'passed')
                           if search_attempt != None else 'after failed attempt'))

                report.step('{0}-cover: {1} #test cases: {0.num_test_cases} '
                            .format(criterion, coverage),
                            '#adversarial examples: {0.num_adversarials} '
                            .format(report),
                            '#diff: {} {}'
                            .format(d if search_attempt != None else '_',
                                    target.log_repr()))

                # 更新迭代次数
                iteration += 1

        except EarlyTermination as e:
            p1('{}'.format(e))
        except KeyboardInterrupt:
            p1('Interrupted.')

        # 运行结束
        p1('Terminating after {} iteration{}: '
           '{} test{} generated, {} of which {} adversarial.'
           .format(*s_(iteration - 1),
                   *s_(report.num_tests - init_tests),
                   *is_are_(report.num_adversarials - init_adversarials)))

        # 打印记录
        if len(record) > 0:
            report.record(criterion.test_cases, record,
                          norm=repr(criterion.metric))

        # 终止
        criterion.terminate()

        # 返回报告文件
        return report

    # 私有函数：状态初始化函数, 初始化覆盖率，约束求解器，上下界等
    def _stat_based_inits(self):
        '''
        Performs basic and incremental static initializations of the
        criterion (and its associated analyzer).
        '''

        objects = [self.criterion,
                   self.criterion.analyzer,
                   self.criterion.analyzer.input_metric] \
                  + self.criterion.analyzer.input_bounds \
                  + self.static_filters \
                  + self.dynamic_filters
        for o in objects:
            if isinstance(o, _InputsStatBasedInitializable):
                o.inputs_stat_initialize(train_data=self.train_data,
                                         test_data=self.ref_data)

        def _acc_initializers(acc, o):
            if isinstance(o, _ActivationStatBasedInitializable):
                acc[0].extend(o.stat_based_basic_initializers())
                acc[1].extend(o.stat_based_incremental_initializers())
                acc[2].extend(o.stat_based_train_cv_initializers())
                acc[3].extend(o.stat_based_test_cv_initializers())
            return acc

        ggi, gi, trcv, tscv = \
            reduce(_acc_initializers, objects, ([], [], [], []))

        # Run stats on batched activations, and/or accumulate for layers
        # that require full activations for their stats.

        if gi == [] and ggi == [] and trcv == [] and tscv == []:
            return

        if gi != [] or ggi != []:
            if gi != []:
                np1('Computing {}... '
                    .format(' & '.join((map((lambda gi: gi['name']), gi)))))
            else:
                np1('Aggregating activations required for {}... '
                    .format(' & '.join((map((lambda gg: gg['name']), ggi)))))
            acc = [None for _ in gi]
            acc_indexes = set().union(*(gi['layer_indexes'] for gi in gi \
                                        if 'layer_indexes' in gi))
            gacc_indexes = set().union(*(gg['layer_indexes'] for gg in ggi))
            gacc = dict.fromkeys(gacc_indexes, np.array([]))
            for act in self._batched_activations_on_raw_data \
                        (layer_indexes=acc_indexes.union(gacc_indexes)):
                acc = [g['accum'](act, acc) for g, acc in zip(gi, acc)]
                if ggi != []:
                    for j in gacc_indexes:
                        gacc[j] = (np.concatenate((gacc[j], act[j]), axis=0)
                                   if gacc[j].any() else np.copy(act[j]))
            for g, acc in zip(gi, acc):
                if 'final' in g: g['final'](acc)
            print('done.')
            for g in gi:
                if 'print' in g: print(g['print']())
            print('', end='', flush=True)

            # Now we can pass the aggregated activations to basic stat
            # initializers.

            if ggi == []:
                return

            np1('Computing {}... '
                .format(' & '.join((map((lambda gg: gg['name']), ggi)))))
            for gg in ggi:
                gg['once']({j: gacc[j] for j in gg['layer_indexes']})
            print('done.')
            for gg in ggi:
                if 'print' in gg: print(gg['print']())
            print(end='', flush=True)

        if trcv != []:
            self._cv_init(trcv, self.train_data)

        if tscv != []:
            self._cv_init(tscv, self.ref_data)

    # 私有函数：用数据初始化覆盖的层激活情况
    def _cv_init(self, cv, data):
        idxs = np.arange(len(data.data))
        for x in cv:
            np1('Computing {}... '.format(x['name']))

            train_size = x['train_size'] if 'train_size' in x else None
            test_size = x['test_size'] if 'test_size' in x else None

            rng = np.random.default_rng(randint())
            if 'train' in x:
                if isinstance(train_size, int) and isinstance(test_size, int):
                    train_size = max(1, min(train_size, len(idxs) - test_size))
                elif isinstance(train_size, int):
                    train_size = min(train_size, len(idxs) - 1)
                elif isinstance(train_size, float):
                    train_size = int(round(train_size * len(idxs)))

            if 'test' in x:
                if isinstance(train_size, int) and test_size is None:
                    test_size = min(len(idxs) - train_size, len(idxs) - 1)
                elif isinstance(train_size, int) and isinstance(test_size, int):
                    test_size = min(test_size, len(idxs) - train_size)
                elif isinstance(test_size, int):
                    test_size = min(test_size, len(idxs) - 1)
                elif train_size is not None and isinstance(train_size, float) and \
                        isinstance(test_size, float):
                    test_size = min(test_size, 1. - train_size)

            train_idxs, test_idxs = \
                train_test_split(idxs, test_size=test_size, train_size=train_size) \
                    if 'train' in x and 'test' in x else \
                    (rng.choice(a=idxs, axis=0, size=min(train_size, len(idxs))), None) \
                        if 'train' in x else \
                        (None, rng.choice(a=idxs, axis=0, size=min(test_size, len(idxs))))

            dnn = self.criterion.analyzer.dnn
            lazy_acts = lazy_activations_on_indexed_data
            if 'train' in x:
                lazy_acts(x['train'], dnn, data, train_idxs, x['layer_indexes'])

            if 'test' in x:
                lazy_acts(x['test'], dnn, data, test_idxs, x['layer_indexes'])

    # 私有方法：批次的激活函数情况
    def _batched_activations_on_raw_data(self, **kwds):
        return self.criterion._batched_activations(self.ref_data.data, **kwds)

    # ---


# ---
# 设置函数，设置engine类
def setup(test_object: test_objectt = None,
          cover_layers: Sequence[CoverableLayer] = None,
          setup_analyzer: Callable[[dict], Analyzer] = None,
          setup_criterion: Callable[[Sequence[CoverableLayer],
                                     Analyzer, dict], Criterion] = None,
          criterion_args: dict = {},
          engine_args: dict = {},
          **kwds) -> Engine:
    """
    Helper to build engine instances.  Extra arguments are passed to the
    analyzer setup function (`setup_analyzer`).

    Note: only fields ``dnn``, ``raw_data``, and ``train_data`` are
    required from `test_object`.

    Extra arguments are passed to `setup_analyzer`.
    """

    print('DNN under test has {0} layer functions, {1} of which {2} to be covered:'
          .format(len(get_layer_functions(test_object.dnn)[0]),
                  *is_are_(len(cover_layers))),
          [cl for cl in cover_layers],
          sep='\n', end='\n\n')
    analyzer = setup_analyzer(analyzed_dnn=test_object.dnn, **kwds)
    criterion = setup_criterion(cover_layers, analyzer, **criterion_args)
    return Engine(test_object.raw_data, test_object.train_data, criterion,
                  **engine_args)


# ------------------------------------------------------------------------------
# Provide slightly more specialized classes:

# 表示一个覆盖层，其中神经元的覆盖与否是使用每个神经元的布尔映射定义的。
class BoolMappedCoverableLayer(CoverableLayer):
    '''
    表示一个覆盖层，其中神经元的覆盖与否是使用每个神经元的布尔映射定义的。
    Represents a layer where coverage is defined using a Boolean mapping
    from each neuron.
    '''

    # 构造函数
    def __init__(self,
                 feature_indices=None,
                 bottom_act_value=MIN,
                 **kwds):
        super().__init__(**kwds)
        self._initialize_map(feature_indices)
        self.activations = []  ## to store some neuron activations
        self.bottom_act_value = bottom_act_value
        self.filtered_out = 0

    # 私有方法：初始化这个映射关系
    def _initialize_map(self, feature_indices) -> None:
        shape = tuple(self.layer.output_shape)
        self.feature_indices = feature_indices
        self.map = np.ones(shape[1:], dtype=bool)
        if self.is_conv and feature_indices != None:
            for i in range(0, self.map.shape[-1]):
                if not i in feature_indices:
                    self.map[..., i] = False

    # 过滤掉padding部分，主要针对卷积层
    def filter_out_padding_against(self, prev_layer):
        if not self.is_conv: return
        tp1(f'Filtering out padding neurons at layer {self}')
        paddings = 0
        for n in np.ndindex(self.map.shape):
            if self.map[n]:
                if is_padding(n, self, prev_layer, post=True, unravel_pos=False):
                    self.map[n] = False
                    paddings += 1
        self.filtered_out += paddings

    # 判断过滤是否有效
    def valid_conv_filters(self):
        if not self.is_conv or \
                self.feature_indices is None or \
                self.feature_indices == []:
            return ()

        fltrs = ()
        for f in self.feature_indices:
            if f >= self.layer.output_shape[-1]:
                p1(f'Ignoring filter {f} for layer {str(self)} '
                   f'with {self.layer.output_shape[-1]} filters')
            else:
                p1(f'Selecting filter {f} for layer {str(self)}')
                fltrs += (f,)
        return fltrs

    # 计算神经元覆盖率
    def coverage(self) -> Coverage:
        if not self.is_conv or self.feature_indices == None:
            nc = np.count_nonzero(self.map)
            tot = self.map.size
        else:
            nc, tot = 0, 0
            for i in range(0, self.map.shape[-1]):
                if not i in self.feature_indices: continue
                nc += np.count_nonzero(self.map[..., i])
                tot += self.map[..., i].size
        tot -= self.filtered_out
        return Coverage(covered=tot - nc, total=tot)

    ## to get the index of the next property to be satisfied
    # [ eq. (15,17,18)? ]
    # 找到下一个满足属性的层
    def find(self, f):
        acts = np.array(self.activations)
        spos = f(acts)
        pos = np.unravel_index(spos, acts.shape)
        return pos, acts.item(spos)

    # 神经元是否被覆盖
    def cover_neuron(self, pos) -> None:
        self.map[pos] = False

    # 禁止在给定位置的任何激活，
    # 这样它就不会被随后对“find”函数的任何直接调用所返回
    # (即在此之前不会调用“update_with_new_activations”)。
    def inhibit_activation(self, pos) -> None:
        '''
        Inhibit any activation at the given position so that it is not
        returned by any direct subsequent call to `find` (i.e not preceded
        by a call to `update_with_new_activations`).
        '''
        act = self.activations
        while len(pos) != 1:
            act = act[pos[0]]
            pos = pos[1:]
        act[pos] = self.bottom_act_value

    # 更新覆盖情况
    def update_with_new_activations(self, acts) -> None:
        for act in acts[self.layer_index]:
            act = np.array([copy.copy(act)])
            # Keep only negative new activation values:
            # TODO: parameterize this (ditto bottom_act_value)
            act[act >= 0] = 0
            self.map = np.logical_and(self.map, act[0])
            # Append activations after map change
            self._append_activations(act)

    # 私有方法：append新的激活函数
    def _append_activations(self, act):
        '''
        Append given activations into the internal buffer.
        '''
        if len(self.activations) >= BUFFER_SIZE:
            self.activations.pop(-1)
        self.activations.insert(0, act)
        self._filter_out_covered_activations()

    # 私有方法：过滤出覆盖的激活函数
    def _filter_out_covered_activations(self):
        for j in range(0, len(self.activations)):
            # Only keep values of non-covered activations
            self.activations[j] = np.multiply(self.activations[j], self.map)
            self.activations[j][self.activations[j] >= 0] = self.bottom_act_value

    # 弹出激活函数情况
    def pop_activations(self):
        self.activations.pop()


# ---

# 用于实现多态的类，主要用于扩展
class LayerLocalAnalyzer(Analyzer):
    '''
    Analyzers that seek layer-local criteria must inherit this class to
    register the sequence of covered layers.
    '''

    @abstractmethod
    def finalize_setup(self, clayers: Sequence[CoverableLayer]):
        raise NotImplementedError


# ---

# 与层相关的覆盖属性需要用在这个类来实现,也是为了实现多态和扩展性
class LayerLocalCriterion(Criterion):
    '''
    Criteria whose definition involves layer-local coverage properties.

    - `shallow_first = True` indicates that shallower layers are given
      priority when selecting new test targets.
    '''

    # 构造函数，初始化相应的覆盖层
    def __init__(self,
                 clayers: Sequence[BoolMappedCoverableLayer] = None,
                 shallow_first=True,
                 **kwds):
        self.shallow_first = shallow_first
        super().__init__(clayers, **kwds)
        for cl in self.cover_layers:
            assert isinstance(cl, BoolMappedCoverableLayer)

    def finalize_setup(self):
        if isinstance(self.analyzer, LayerLocalAnalyzer):
            self.analyzer.finalize_setup(self.cover_layers)

    @property
    def _updatable_layers(self):
        '''
        Gives the set of all internal objects that are updated upon
        insertion of a new test case.
        '''
        return self.cover_layers

    def stat_based_incremental_initializers(self):
        if len(self.cover_layers) <= 1:
            for cl in self.cover_layers: cl.pfactor = 1.0
            return []
        else:
            return [{
                'name': 'magnitude coefficients',
                'accum': self._acc_magnitude_coefficients,
                'final': self._calculate_pfactors,
                'print': (lambda: [cl.pfactor for cl in self.cover_layers]),
                'layer_indexes': set([cl.layer_index for cl in self.cover_layers])
            }]

    def _acc_magnitude_coefficients(self, new_acts, acc=None):
        acc = acc or dict()
        for cl in self.cover_layers:
            cl_acts = new_acts[cl.layer_index]
            cl_acts_average = np.average(np.abs(cl_acts))
            if cl.layer_index not in acc:
                acc[cl.layer_index] = (cl_acts_average, len(cl_acts))
            else:
                pavg, plen = acc[cl.layer_index]
                nlen = plen + len(cl_acts)
                navg = (cl_acts_average * len(cl_acts) + pavg * plen) / nlen
                acc[cl.layer_index] = (navg, nlen)
        return acc

    def _calculate_pfactors(self, acc):
        fks = [acc[cl.layer_index][0] for cl in self.cover_layers]
        av = np.average(fks)
        for cl, fks in zip(self.cover_layers, fks):
            cl.pfactor = av / fks

    # ---

    # 返回计算的覆盖率
    def coverage(self) -> Coverage:
        c = Coverage(total=0)
        for cl in self.cover_layers:
            # if self.test_object.tests_layer (cl):
            # assert (self.test_object.tests_layer (cl))
            c += cl.coverage()
        return c

    # ---

    # 弹出最有一个插入的测试用例并根据新的测试target更新记录
    def pop_test(self):
        '''
        Pop last inserted test case, and update the associated recorded
        activations used to find new test targets.
        '''
        super().pop_test()
        for cl in self._updatable_layers:
            cl.pop_activations()

    # 注册新的测试用例
    def register_new_activations(self, acts):
        """
        Register new test cases
        """
        for cl in self._updatable_layers:
            cl.update_with_new_activations(acts)

    # 获取最大的覆盖层序号
    def get_max(self) -> Tuple[BoolMappedCoverableLayer, Tuple[int, ...], float, Input]:
        '''
        '''
        layer, pos, value = None, None, MIN
        for i, cl in enumerate(self.cover_layers):
            p, v = cl.find(np.argmax)
            v *= cl.pfactor
            if v > value:
                layer, pos, value = i, p, v
                if self.shallow_first: break
                if np.random.uniform(0., 1.) < i * 1.0 / len(self.cover_layers): break
        if layer == None:
            raise EarlyTermination('Unable to find a new candidate input!')
        return self.cover_layers[layer], pos, value, self.test_cases[-1 - pos[0]]

    # 获取随机的覆盖层序号
    def get_random(self):
        clx = [cl for cl in self.cover_layers if np.any(cl.map)]
        if clx == []:
            return None
        else:
            while True:
                idx = np.random.randint(0, len(clx))
                cl = clx[idx]
                tot_s = cl.map.size
                pos = (np.random.randint(0, tot_s) if cl.feature_indices is None else \
                           np.argmax(cl.map.shape))
                while pos < tot_s and not cl.map.item(pos):
                    pos += 1
                if pos < tot_s and cl.map.item(pos):
                    break
        return cl, (0,) + np.unravel_index(pos, cl.map.shape)

# ---
