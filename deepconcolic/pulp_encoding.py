from utils_io import *
from utils import *
from pulp import *
from engine import Bounds

lt_epsilon = 1e-5
act_epsilon = 1e-4


# ---

# 符号化每一层
class PulpLayerOutput:
    """
    Abstract representation to obtain symbolic expressions that encode
    layer outputs.
    """

    @abstractmethod
    def pulp_gen_vars(self, idx: int, var_names: dict) -> int:
        raise NotImplementedError

    @abstractmethod
    def pulp_out_exprs(self):
        raise NotImplementedError


# ---

# 将符号化的层encoding
class BasicPulpInputLayerEncoder(PulpLayerOutput):
    """
    Input layer encoder for pulp.
    """
    # 构造函数
    def __init__(self, shape=None, bounds: Bounds = None, **kwds):
        assert shape is not None and isinstance(shape, (tuple, tf.TensorShape))
        assert bounds is not None and isinstance(bounds, Bounds)
        self.shape = shape
        self.bounds = bounds
        super().__init__(**kwds)

    def pulp_gen_vars(self, idx: int, var_names: dict) -> int:
        new_idx = gen_vars(idx, self.shape, var_names)
        self.var_names = var_names[idx]
        return new_idx

    def pulp_in_vars(self):
        return self.var_names[0]

    def pulp_out_exprs(self):
        return self.var_names

    def pulp_bounds(self, name_prefix='input_') -> Sequence[LpConstraint]:
        cstrs = []
        for idx, x in np.ndenumerate(self.var_names):
            # NB: `vname` is only used for identifying coinstraints
            vname = '_'.join(str(i) for i in (name_prefix,) + idx)
            low, up = self.bounds[idx[1:]]

            cstrs.extend([
                # x<=ub
                LpConstraint(LpAffineExpression([(x, +1)]),
                             LpConstraintLE, rhs=up,
                             name='{}<=ub'.format(vname)),

                # x>=lb
                LpConstraint(LpAffineExpression([(x, +1)]),
                             LpConstraintGE, rhs=low,
                             name='{}>=lb'.format(vname))
            ])
        return cstrs


# ---

# 通用的pulp编码器
class PulpLayerEncoder(PulpLayerOutput):
    """
    Generic layer encoder for pulp.
    """

    @abstractmethod
    def pulp_gen_base_constraints(self, base_prob: LpProblem, base_prob_dict: dict,
                                  prev: PulpLayerOutput) -> None:
        raise NotImplementedError


# ---

# 满足一定条件的pulp编码器
class PulpStrictLayerEncoder(PulpLayerEncoder, PulpLayerOutput):

    def __init__(self, l, layer, nonact_layers=False, **kwds):
        super().__init__(**kwds)
        self.layer_index = l
        self.layer = layer
        self.nonact_layers = nonact_layers

    def pulp_gen_vars(self, idx, var_names):
        layer = self.layer
        tp1('Creating base variables for layer {} ({})'.format(layer.name, self.layer_index))

        ## create variables for layer OUTPUT neurons
        if is_input_layer(layer):
            # already done: just check the input shape?
            if len(layer.input.shape) <= 2:
                sys.exit('We assume the input layer to be conv...')

        elif is_conv_layer(layer):
            idx = gen_vars(idx, layer.output.shape, var_names)
            self.u_var_names = var_names[-1]
            if activation_is_relu(layer):  # 'conv+relu'
                idx = gen_vars(idx, layer.output.shape, var_names)

        elif is_dense_layer(layer):
            idx = gen_vars(idx, layer.output.shape, var_names)
            self.u_var_names = var_names[-1]
            if activation_is_relu(layer):  # 'dense+relu'
                idx = gen_vars(idx, layer.output.shape, var_names)

        elif is_activation_layer(layer):
            if not activation_is_relu(layer):
                p1('Assuming {} is ReLU ({})'.format(layer.name, self.layer_index))
            idx = gen_vars(idx, layer.output.shape, var_names)

        elif is_maxpooling_layer(layer) or is_dropout_layer(layer):
            idx = gen_vars(idx, layer.output.shape, var_names)

        elif is_flatten_layer(layer) or is_reshape_layer(layer):
            pass

        else:
            self._unknown()

        self.output_var_names = var_names[-1]
        return idx

    def _unknown(self):
        sys.exit(f'Unknown layer {self.layer.name} at index {self.layer_index} '
                 f'({type(self.layer)})')

    def pulp_out_exprs(self):
        if is_flatten_layer(self.layer) or is_reshape_layer(self.layer):
            out_shape = tuple(d or 1 for d in self.layer.output.shape)
            return self.output_var_names.reshape(out_shape)
        else:
            return self.output_var_names

    def pulp_gen_base_constraints(self, base_prob, base_prob_dict, prev):
        layer = self.layer
        tp1('Creating base constraints for layer {} ({})'
            .format(layer.name, self.layer_index))

        assert isinstance(prev, PulpLayerOutput)
        in_exprs = prev.pulp_out_exprs()
        out_vars = self.output_var_names
        isp = in_exprs.shape

        if is_input_layer(layer):
            ## nothing to constrain for InputLayer
            pass

        elif is_conv_layer(layer):
            u_vars = self.u_var_names
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            for nidx in np.ndindex(u_vars.shape):
                u_var = u_vars[nidx]
                affine_expr = [(u_var, -1)]
                for kidx in np.ndindex(layer.kernel_size):
                    for KK in range(0, weights.shape[-1]):
                        try:
                            in_expr = in_exprs[0][nidx[1] + kidx[0]][nidx[2] + kidx[1]][KK]
                            affine_expr.append((in_expr, float(weights[kidx][KK][nidx[-1]])))
                        except:
                            ## padding
                            pass
                base_prob += LpConstraint(LpAffineExpression(affine_expr),
                                          LpConstraintEQ,
                                          'c_name_conv_{0}'.format(u_var),
                                          -float(biases[nidx[-1]]))

            if self.nonact_layers or activation_is_relu(layer):
                base_prob_dict[self.layer_index] = base_prob.copy()

        elif is_dense_layer(layer):
            u_vars = self.u_var_names
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            for nidx in np.ndindex(u_vars.shape):
                u_var = u_vars[nidx]
                affine_expr = [(u_var, -1)]
                for II in range(0, isp[-1]):
                    affine_expr.append((in_exprs[0][II], float(weights[II][nidx[-1]])))
                base_prob += LpConstraint(LpAffineExpression(affine_expr),
                                          LpConstraintEQ,
                                          'c_name_dense_{0}'.format(u_var),
                                          -float(biases[nidx[-1]]))

            if self.nonact_layers or activation_is_relu(layer):
                base_prob_dict[self.layer_index] = base_prob.copy()

        elif is_maxpooling_layer(layer):
            pool_size = layer.pool_size
            assert (pool_size == layer.strides)  # in case
            assert not is_activation_layer(layer)
            for oidx in np.ndindex(out_vars.shape[1:]):
                out_var = out_vars[0][oidx]
                for poolidx in maxpool_idxs(oidx, pool_size):
                    cname = '_'.join(str(i) for i in ("mpcr__", self.layer_index,) +
                                     oidx + poolidx)
                    c = LpAffineExpression([(out_var, +1),
                                            (in_exprs[0][poolidx][oidx[-1]], -1)])
                    base_prob += LpConstraint(c, LpConstraintGE, cname, 0.)
            if self.nonact_layers:
                base_prob_dict[self.layer_index] = base_prob.copy()

        elif is_dropout_layer(layer):
            p = float(1. - layer.rate)
            for nidx in np.ndindex(in_exprs.shape):
                u_var = in_exprs[nidx]
                affine_expr = [(out_vars[nidx], 1), (u_var, -p)]
                base_prob += LpConstraint(LpAffineExpression(affine_expr),
                                          LpConstraintEQ,
                                          'c_name_dropout_{0}'.format(u_var),
                                          0.0)
            if self.nonact_layers:
                base_prob_dict[self.layer_index] = base_prob.copy()

        elif is_flatten_layer(layer) or is_reshape_layer(layer):
            pass

        elif is_activation_layer(layer):  # Assuming ReLU activation
            base_prob_dict[self.layer_index] = base_prob.copy()

        else:
            self._unknown()

    def pulp_replicate_behavior(self, ap_x, prev: PulpLayerOutput) -> Sequence[LpConstraint]:
        layer = self.layer
        if not is_maxpooling_layer(layer):
            return []

        u_exprs = prev.pulp_out_exprs()
        v_vars = self.output_var_names
        cstrs = []

        if is_maxpooling_layer(layer):
            pool_size = layer.pool_size
            for oidx in np.ndindex(v_vars.shape[1:]):
                oap = ap_x[self.layer_index][0][oidx]
                v_var = v_vars[0][oidx]
                for poolidx in maxpool_idxs(oidx, pool_size):
                    if oap == ap_x[self.layer_index - 1][0][poolidx][oidx[-1]]:
                        cname = '_'.join(str(i) for i in ("mpcn__", self.layer_index,) +
                                         oidx + poolidx)
                        c = LpAffineExpression([(v_var, +1),
                                                (u_exprs[0][poolidx][oidx[-1]], -1)])
                        cstrs.append(LpConstraint(c, LpConstraintEQ, cname, 0.))

        else:
            self._unknown()

        return cstrs

    def pulp_replicate_activations(self, ap_x, prev: PulpLayerOutput,
                                   exclude=(lambda _: False)) -> Sequence[LpConstraint]:
        layer = self.layer
        if (is_input_layer(layer) or
                is_flatten_layer(layer) or
                is_reshape_layer(layer) or
                is_dropout_layer(layer)):
            return []

        elif (is_conv_layer(layer) or is_dense_layer(layer) or is_activation_layer(layer)):
            constrain_output = not (is_conv_layer(layer) and not activation_is_relu(layer) or
                                    is_dense_layer(layer) and not activation_is_relu(layer))
            u_exprs = prev.pulp_out_exprs() if is_activation_layer(layer) else self.u_var_names
            v_vars = self.output_var_names if constrain_output else None
            v_idx = self.layer_index - 1 if is_activation_layer(layer) else self.layer_index
            cstrs = []
            for oidx in np.ndindex(self.output_var_names.shape):
                if exclude(oidx): continue
                cstrs.extend(same_act(self.layer_index, v_vars, u_exprs, oidx, ap_x[v_idx]))
            return cstrs

        elif is_maxpooling_layer(layer):
            return self.pulp_replicate_behavior(ap_x, prev)

        else:
            self._unknown()

    def pulp_negate_activation(self, ap_x, oidx,
                               prev: PulpLayerOutput) -> Sequence[LpConstraint]:
        layer = self.layer
        assert not is_input_layer(layer)
        assert not is_flatten_layer(layer)

        u_exprs = prev.pulp_out_exprs()
        v_vars = self.output_var_names
        cstrs = []

        if (is_conv_layer(layer) or is_dense_layer(layer) or is_activation_layer(layer)):
            constrain_output = not (is_conv_layer(layer) and not activation_is_relu(layer) or
                                    is_dense_layer(layer) and not activation_is_relu(layer))
            u_exprs = u_exprs if is_activation_layer(layer) else self.u_var_names
            v_vars = v_vars if constrain_output else None
            v_idx = self.layer_index - 1 if is_activation_layer(layer) else self.layer_index
            cstrs.extend(neg_act(self.layer_index, v_vars, u_exprs, oidx, ap_x[v_idx]))

        elif is_maxpooling_layer(layer):
            # XXX: Ignoring oidx and constrain activation of max.
            assert False
            # pool_size = layer.pool_size
            # for oidx in np.ndindex (v_vars.shape):
            #   max_found = False
            #   for II in range(oidx[0] * pool_size[0], (oidx[0] + 1) * pool_size[0]):
            #     for JJ in range(oidx[1] * pool_size[1], (oidx[1] + 1) * pool_size[1]):
            #       if not max_found and (ap_x[self.layer_index][0][oidx] ==
            #                             ap_x[self.layer_index - 1][0][II][JJ][oidx[2]]):
            #         max_found = True
            #         cname = '_'.join(str(i) for i in ("mpcn__", self.layer_index, ) + oidx + (II, JJ,))
            #         c = LpAffineExpression([(v_vars[0][oidx], +1),
            #                                 (u_exprs[0][II][JJ][oidx[2]], -1)])
            #         cstrs.append(LpConstraint(c, LpConstraintEQ, cname, 0.))

        else:
            self._unknown()

        return cstrs


# ---


strict_encoder = PulpStrictLayerEncoder

# 设置层编码器
def setup_layer_encoders(dnn, build_encoder, input_bounds: Bounds,
                         first=0, upto=None):
    upto = -1 if upto == None else max(-1, upto + 1)
    lc, var_names = [], []
    ## Create variables for INPUT neurons
    ilc = BasicPulpInputLayerEncoder \
        (shape=dnn.layers[first].input.shape, bounds=input_bounds)
    idx = ilc.pulp_gen_vars(0, var_names)
    for l, layer in enumerate(dnn.layers[first:upto]):
        lcl = build_encoder(l, layer)
        assert isinstance(lcl, PulpLayerEncoder)
        idx = lcl.pulp_gen_vars(idx, var_names)
        lc.append(lcl)
    return lc, ilc, var_names


# ---

# 创建约束
def create_base_problem(layer_encoders, input_layer_encoder):
    base_prob = LpProblem("base_prob", LpMinimize)
    base_prob.extend(input_layer_encoder.pulp_bounds())
    base_prob_dict = dict()
    prev = input_layer_encoder
    for l in layer_encoders:
        l.pulp_gen_base_constraints(base_prob, base_prob_dict, prev)
        prev = l
    return base_prob_dict

# 生成变量
def gen_vars(layer_index, sp, var_names, flatten=False):
    shape = (1,) + ((np.prod(sp[1:]),) if flatten else tuple(sp[1:]))
    var_names.append(np.empty(shape, dtype=LpVariable))
    for idx in np.ndindex(*shape):
        var = LpVariable('_'.join(str(i) for i in ("x", layer_index) + idx))
        var_names[layer_index][idx] = var
    return layer_index + 1


# ---

# 返回一组约束，这些约束在位置“pos”((0)，+ neuron)处重现激活模式“ap_x”
# 基于约束求解的变量和表达式
def same_act(base_name, v_vars, u_exprs, pos, ap_x):
    """
    Returns a set of constraints that reproduces the activation pattern
    `ap_x` at position `pos` ((0,) + neuron), based on:

    - pulp variables `v_vars` that encode neuron outputs; and

    - pulp expressions `u_exprs` that encode the pre-activation value of
      each neuron.
    """

    cname = '_'.join(str(i) for i in ("sa__", base_name,) + pos)

    if ap_x[pos] > 0:
        x = [LpConstraint(LpAffineExpression([(v_vars[pos], +1), (u_exprs[pos], -1)]),
                          LpConstraintEQ, cname + '_eq', 0.)] if v_vars is not None else []
        x += [LpConstraint(LpAffineExpression([(u_exprs[pos], +1)]),
                           LpConstraintGE, cname + '_ge', float(act_epsilon))]
        return x
    else:
        x = [LpConstraint(LpAffineExpression([(v_vars[pos], +1)]),
                          LpConstraintEQ, cname + '_eq', 0.)] if v_vars is not None else []
        x += [LpConstraint(LpAffineExpression([(u_exprs[pos], +1)]),
                           LpConstraintLE, cname + '_le', float(-act_epsilon))]
        return x

# 返回一组约束，这些约束在位置“pos”((0)，+ neuron)处否定了激活模式“ap_x”
# 基于约束求解的变量和表达式
def neg_act(base_name, v_vars, u_exprs, pos, ap_x):
    """
    Returns a set of constraints that negates the activation pattern
    `ap_x` at position `pos` ((0,) + neuron), based on:

    - pulp variables `v_vars` that encode neuron outputs; and

    - pulp expressions `u_exprs` that encode the pre-activation value of
      each neuron.
    """

    cname = '_'.join(str(i) for i in ("na__", base_name,) + pos)

    if ap_x[pos] <= 0:
        x = [LpConstraint(LpAffineExpression([(v_vars[pos], +1), (u_exprs[pos], -1)]),
                          LpConstraintEQ, cname + '_eq', 0.)] if v_vars is not None else []
        x += [LpConstraint(LpAffineExpression([(u_exprs[pos], +1)]),
                           LpConstraintGE, cname + '_ge', float(act_epsilon))]
        return x
    else:
        x = [LpConstraint(LpAffineExpression([(v_vars[pos], +1)]),
                          LpConstraintEQ, cname + '_eq', 0.)] if v_vars is not None else []
        x += [LpConstraint(LpAffineExpression([(u_exprs[pos], +1)]),
                           LpConstraintLE, cname + '_le', float(-act_epsilon))]
        return x


# Those are now just aliases:

# 创建卷积层约束
def build_conv_constraint(base_name, v_vars, u_exprs, pos, ap_x):
    """Alias for :func:`same_act`"""
    return same_act(base_name, v_vars, u_exprs, pos, ap_x)

# 创建全连接层约束
def build_dense_constraint(base_name, v_vars, u_exprs, pos, ap_x):
    """Alias for :func:`same_act`"""
    return same_act(base_name, v_vars, u_exprs, pos, ap_x)

# 创建卷积否定约束
def build_conv_constraint_neg(base_name, v_vars, u_exprs, pos, ap_x):
    """Alias for :func:`neg_act`"""
    return neg_act(base_name, v_vars, u_exprs, pos, ap_x)

# 创建全连接层否定约束
def build_dense_constraint_neg(base_name, v_vars, u_exprs, pos, ap_x):
    """Alias for :func:`neg_act`"""
    return neg_act(base_name, v_vars, u_exprs, pos, ap_x)
