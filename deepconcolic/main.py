#!/usr/bin/env python4
import argparse
from pathlib import Path
from utils_io import *
from utils_funcs import *
from utils import *
from bounds import UniformBounds, StatBasedInputBounds
import datasets
import yaml


def deepconcolic(criterion, norm, test_object, report_args,
                 engine_args={},
                 norm_args={},
                 dbnc_spec={},
                 input_bounds=None,
                 postproc_inputs=id,
                 run_engine=True,
                 **engine_run_args):
    test_object.check_layer_indices(criterion)
    engine = None
    # test_engine_run_args = {}
    # test_engine_run_args.update(**engine_run_args)
    # print(test_engine_run_args)
    # exit(0)
    # 当标准是neuron coverage的时候
    if criterion == 'nc':  ## neuron cover
        # 从nc文件中导入setup作为nc_setup
        from nc import setup as nc_setup
        # 如果采用的范式是L-infinity, 即无穷范式，也称切比雪夫距离，取最大值的范式
        if norm == 'linf':
            # 从pulp文件中导入LInfPulp
            from pulp_norms import LInfPulp
            # 从nc_pulp文件中导入NcPulpAnalyzer，这里的pulp不知道什么意思，但是ncpulpanalyzer的作用应该是分析神经元覆盖率
            from nc_pulp import NcPulpAnalyzer
            # 初始化engine, 用于求解路径并计算nc
            engine = nc_setup(test_object=test_object,
                              engine_args=engine_args,
                              setup_analyzer=NcPulpAnalyzer,
                              input_metric=LInfPulp(**norm_args),
                              input_bounds=input_bounds,
                              postproc_inputs=postproc_inputs)
        else:
            print('\n not supported norm... {0}\n'.format(norm))
            sys.exit(0)

    # 初始化好engine, 直接run, run的过程中会调用其他文件完成优化
    if engine != None and run_engine:
        return engine, engine.run(**report_args, **engine_run_args)
    # 返回engine, 不过没有接收部分，主要用于可能的拓展
    return engine


def concolic_main(model_path_from_run, generated_images_path, generated_label_of_images_path, concolic_output_path):
    # 这些参数用于扩展命令行模式用
    parser = argparse.ArgumentParser \
        (description='Concolic testing for Neural Networks',
         prog='python3 -m deepconcolic.main',
         prefix_chars='-+')

    parser.add_argument('--norm-factor', metavar='FLOAT', type=float, default=1,
                        help='norm distance upper threshold above which '
                             'generated inputs are rejected by the oracle (default is 1/4)')
    parser.add_argument('--lb-hard', metavar='FLOAT', type=float,
                        help='hard lower bound for the distance between '
                             'original and generated inputs (concolic engine only---'
                             'default is 1/255 for image datasets, 1/100 otherwise)')
    parser.add_argument('--lb-noise', metavar='FLOAT', type=float,
                        help='extra noise on the lower bound for the distance '
                             'between original and generated inputs (concolic engine '
                             'only---default is 1/10)')
    parser.add_argument("--mcdc-cond-ratio", dest="mcdc_cond_ratio", metavar="FLOAT",
                        type=float, default=0.01,
                        help="the condition feature size parameter (0, 1]")
    parser.add_argument("--top-classes", dest="top_classes", metavar="CLS",
                        type=int, default=1,
                        help="check the top-CLS classifications for models that "
                             "output estimations for each class (e.g. VGG*)")
    parser.add_argument("--layers", dest="layers", nargs="+", metavar="LAYER",
                        help="test layers given by name or index")
    parser.add_argument("--feature-index", dest="feature_index", default="-1",
                        help="to test a particular feature map", metavar="INT")
    #
    # DBNC-specific params
    parser.add_argument("--dbnc-spec", default="{}",
                        help="Feature extraction and discretisation specification",
                        metavar="SPEC")
    args = parser.parse_args()

    # 这个随机种子是为了保证每次的运行结果一致
    rng_seed_given = 43
    try:
        rng_seed(rng_seed_given)
    except ValueError as e:
        sys.exit("Invalid argument given for `--rng-seed': {}".format(e))

    # 不知道有什么用的参数
    inp_ub = 1
    lower_bound_metric_hard = None
    lower_bound_metric_noise = None

    # 通过dataset_dict()获取传入的参数，这里以fashion_mnist为例
    # dataset = "selfdriver"
    dd = dataset_dict(generated_images_path, generated_label_of_images_path)

    # 训练数据，测试数据，种类，存储输入，打印处理过后的输入，输入范围的获取，其中save_input, postproc_inputs, ib不知道有什么用
    train_data, test_data, kind, save_input, postproc_inputs, ib = \
        dd['train_data'], dd['test_data'], dd['kind'], \
        dd['save_input'], dd['postproc_inputs'], dd['input_bounds']
    amplify_diffs = kind in datasets.image_kinds
    if kind in datasets.image_kinds:  # assume 256 res.
        lower_bound_metric_hard = 1 / 255
    input_bounds = (UniformBounds(*ib) if isinstance(ib, tuple) and len(ib) == 2 else \
                        StatBasedInputBounds(hard_bounds=UniformBounds(-1.0, 1.0)) \
                            if ib == 'normalized' else StatBasedInputBounds())
    del dd

    # input_filters，拓展参数，暂时没有作用
    input_filters = []

    # 通过load_model加载传入的模型参数
    model_path_from_run = model_path_from_run
    dnn = load_model(model_path_from_run)
    # dnn.summary可以把模型的输出维度展示出来
    dnn.summary()

    # 边界值和噪声
    lower_bound_metric_hard = some(lower_bound_metric_hard, 1 / 100)
    lower_bound_metric_noise = some(lower_bound_metric_noise, 1 / 10)
    input_bounds = some(input_bounds, UniformBounds(0.0, 1.0))
    postproc_inputs = some(postproc_inputs, id)

    # 测试对象，通过test_objectt函数，传入的参数分别为，模型dnn,训练数据train_data, 测试数据test_data
    test_object = test_objectt(dnn, train_data, test_data)
    # test_object.cond_ratio = args.mcdc_cond_ratio
    test_object.postproc_inputs = postproc_inputs
    # NB: only used in run_ssc.run_svc (which is probably broken) >>
    test_object.top_classes = int(args.top_classes)
    test_object.inp_ub = inp_ub

    # 指定第一个activation层
    # layers_given = '1'
    # layers_given = '5'
    layers_given = None
    if layers_given is not None:
        try:
            test_object.set_layer_indices(int(l) if l.isdigit() else l
                                          for l in layers_given)
            # print("***********************")
            # print(test_object.layer_indices)
            # exit(0)
            # print("***********************")
        except ValueError as e:
            sys.exit(e)
    if args.feature_index != '-1':
        test_object.feature_indices = [int(args.feature_index)]
        print('feature index specified:', test_object.feature_indices)

    # init_tests = int (args.init_tests) if args.init_tests is not None else None
    # max_iterations = int (args.max_iterations)

    # init_tests = 3
    # max_iterations = 5
    init_tests = 3
    max_iterations = 5

    # DBNC-specific parameters:
    try:
        if args.dbnc_spec != "{}" and os.path.exists(args.dbnc_spec):
            with open(args.dbnc_spec, 'r') as f:
                dbnc_spec = yaml.safe_load(f)
        else:
            dbnc_spec = yaml.safe_load(args.dbnc_spec)
        if len(dbnc_spec) > 0:
            print("DBNC Spec:\n", yaml.dump(dbnc_spec), sep='')
    except yaml.YAMLError as exc:
        sys.exit(exc)

    # 固定参数传入
    criterion = "nc"
    norm = "linf"
    output_path = concolic_output_path
    save_all_tests = True
    norm_factor = 1
    deepconcolic(criterion, norm, test_object,
                 report_args={'outdir': OutputDir(output_path, log=True),
                              'save_new_tests': save_all_tests,
                              'save_input_func': save_input,
                              'amplify_diffs': amplify_diffs},
                 norm_args={'factor': norm_factor,
                            'LB_hard': lower_bound_metric_hard,
                            'LB_noise': lower_bound_metric_noise},
                 engine_args={'custom_filters': input_filters},
                 dbnc_spec=dbnc_spec,
                 input_bounds=input_bounds,
                 postproc_inputs=postproc_inputs,
                 run_engine=True,
                 initial_test_cases=init_tests,
                 max_iterations=max_iterations)


if __name__ == "__main__":
    try:
        model_path_from_run = r"F:\DL_TEST\ai-test-master\ai-test\deepconcolic\saved_models\fashion_mnist_medium.h5"
        # model_path_from_run = r"F:\601\software\ai-master-20211220\ai-test\deepconcolic\saved_models\mnist_medium.h5"
        # model_path_from_run = r"F:\DL_TEST\ai-test-master\ai-test\deepconcolic\saved_models\AlexNet_model.h5"
        # generated_images_path = r"F:\DL_TEST\ai-test-master\ai-test\Dataset\hmb2_100"
        # generated_images_path = "cifar10"
        generated_images_path = "mnist"
        # generated_images_path = "cifar10"
        generated_label_of_images_path = r"F:\DL_TEST\ai-test-master\ai-test\Dataset\hmb2_100\hmb2_steering.csv"
        concolic_output_path = r"F:\Chris_DL\concolic_result"
        concolic_main(model_path_from_run, generated_images_path, generated_label_of_images_path, concolic_output_path)
    except KeyboardInterrupt:
        sys.exit('Interrupted.')
