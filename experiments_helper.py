"""Shared configuration, training, evaluation, and plotting helpers for
the experiment notebooks (regular_experiments.ipynb, etc.)."""

from itertools import product
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from linear_transformer import (
    Transformer_F,
    Transformer_F_Looped,
    generate_data,
    in_context_loss,
    DataMode
)


####################################
# Problem parameters
####################################

# Use CUDA when available, otherwise CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
alg = 'adam'

normal_data_mode = DataMode("normal")

N = 20     # context length (no assumption given in paper)
d = 5        # dimension

n_head = 1  # 1-headed attention
B = 4000  # 4000 minibatch size
var = 0.0001  # initializations scale of transformer parameter
max_iters = 10000  # Number of Iterations to run
hist_stride = 1  # stride for saved model paramters in `train.ipynb'
stride = 100

TEST_SEED = 99

MODEL_STYLES = {
    'layered': ('-', lambda count: f'baseline (L = {count})'),
    'looped': ('--', lambda count: f'looping (l=1, loop={count})'),
}


####################################
# Model construction and training
####################################

def build_transformer(model_type, count, head_count, d, var, attention_type='linear', learn_u=False):
    """Construct the requested architecture with linear or softmax attention.

    `learn_u=True` (looped models only) adds the learnable bias vector u, the
    bottom row of the value matrix in Gatmiry et al. (2024). It defaults to False (u = 0).
    """
    if attention_type == 'linear':
        activation = None
    elif attention_type == 'softmax':
        activation = torch.nn.Softmax(dim=2)
    else:
        raise ValueError(f'Unsupported attention type: {attention_type}')

    if model_type == 'layered':
        model_class = Transformer_F
    elif model_type == 'looped':
        model_class = Transformer_F_Looped
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    if learn_u:
        if model_type != 'looped':
            raise ValueError('learn_u is only supported for looped models')
        return model_class(count, head_count, d, var, activation=activation, learn_u=True)
    return model_class(count, head_count, d, var, activation=activation)


def step(params, optimizer, clip_r=None):
    """Take an optimizer step, optionally clipping the gradient norm first.

    `params` is one parameter tensor (e.g. `model.allparam`) or a list of them
    (e.g. `[model.allparam, model.u]`); the norm is taken jointly over all of them.
    With `clip_r=None` (default) no clipping is applied. Otherwise the gradients
    are rescaled in place so their joint norm is at most `clip_r`. The step is skipped if
    the gradient norm is not finite. Returns the pre-clipping gradient norm.
    """
    if isinstance(params, torch.Tensor):
        params = [params]
    grads = [param.grad for param in params if param.grad is not None]
    norm_p = torch.sqrt(sum(grad.pow(2).sum() for grad in grads)).item()

    if clip_r is not None:
        if norm_p > clip_r:
            for grad in grads:
                grad.mul_(clip_r / norm_p)

    if np.isfinite(norm_p):
        optimizer.step()

    return norm_p


def run_experiment_suite(seeds, n_layers_loops, model_types, filename, context_lengths=None,
                         data_modes: list[DataMode] = None, head_counts: list[int] | None = None,
                         attention_types: list[str] | None = None, clip=1000,
                         freeze_p=False,  # [P-FIX] added freeze_p argument
                         learn_u=False):
    """
    Train every requested run and save all histories to `filename`.

    `clip=True` clips gradient norms at the module-level `clip_r`; pass
    `clip=False` to train without clipping.

    [P-FIX] `freeze_p=True` zeroes and freezes the learnable top-left d x d block
    of P, giving the Gatmiry et al. (2024) parameterization P = [[0, 0], [0, 1]]
    (u = 0 always holds because the model has no u parameter). Only Q (= A) is
    trained. The default `False` keeps the earlier behavior, where that block is trained.

    `learn_u=True` (looped models only) also trains the bias vector u, the bottom
    row of the value matrix (P = [[0, 0], [u^T, 1]] when combined with freeze_p=True).
    Snapshots of u are recorded at the same stride as allparam, saved under
    'u_hist_dict' in `filename`, and the function then returns
    `(hist_dict, u_hist_dict)` instead of only `hist_dict`.

    `context_lengths` defaults to [N], and `data_modes` defaults to [normal_data_mode].
    `head_counts` defaults to [n_head]. `attention_types` lists the attention types
    to train ('linear' and/or 'softmax'); it defaults to ['linear']. Linear runs keep
    the six-field key, other attention types append it as a seventh key field.
    """
    if context_lengths is None:
        context_lengths = [N]
    else:
        context_lengths = list(context_lengths)

    if data_modes is None:
        training_modes = [normal_data_mode]
    else:
        training_modes = list(data_modes)

    if head_counts is None:
        head_counts = [n_head]
    else:
        head_counts = list(head_counts)

    if attention_types is None:
        attention_types = ['linear']
    elif isinstance(attention_types, str):
        attention_types = [attention_types]
    else:
        attention_types = list(attention_types)
    for attention_type in attention_types:
        if attention_type not in {'linear', 'softmax'}:
            raise ValueError(f'Unsupported attention type: {attention_type}')

    base_keys = [(seed, count, model_type, context_length, data_mode, head_count)
                 for seed in seeds
                 for count in n_layers_loops
                 for model_type in model_types
                 for context_length in context_lengths
                 for data_mode in training_modes
                 for head_count in head_counts]

    # Keep the existing six-field key format for linear runs; tag softmax runs
    # so their histories remain distinguishable and compatible with existing plots.
    keys = [key if attention_type == 'linear' else key + (attention_type,)
            for key in base_keys
            for attention_type in attention_types]

    if learn_u and any(model_type != 'looped' for model_type in model_types):
        raise ValueError('learn_u is only supported for looped models')

    hist_dict = {}
    u_hist_dict = {}
    for run_index, key in enumerate(keys, start=1):
        seed, count, model_type, context_length, data_mode, head_count = key[:6]
        attention_type = _attention_type(key)
        print(f'Run {run_index}/{len(keys)}: seed={seed}, count={count}, type={model_type}, '
              f'N={context_length}, mode={data_mode}, heads={head_count}, attention={attention_type}, '
              f'freeze_p={freeze_p}, learn_u={learn_u}')  # [P-FIX] log freeze_p
        hist_dict[key] = []
        if learn_u:
            u_hist_dict[key] = []

        torch.manual_seed(seed)
        model = build_transformer(
            model_type, count, head_count, d, var, attention_type=attention_type,
            learn_u=learn_u,
        ).to(device)
        step_params = [model.allparam, model.u] if learn_u else model.allparam

        # [P-FIX] Zero the random initialization of P's top-left block.
        if freeze_p:
            model.zero_p()

        if alg == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        elif alg == 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=0)
        else:
            raise ValueError(f'Unsupported optimizer: {alg}')

        np.random.seed(seed)
        torch.manual_seed(seed)
        for t in range(max_iters):
            start = time.time()
            if t % hist_stride == 0:
                hist_dict[key].append(model.allparam.detach().clone())
                if learn_u:
                    u_hist_dict[key].append(model.u.detach().clone())

            Z, y = generate_data(data_mode, context_length, d, B)
            Z = Z.to(device)
            y = y.to(device)
            loss = in_context_loss(model, Z, y)

            loss.backward()

            # [P-FIX] Drop the P gradient so the optimizer never updates P.
            if freeze_p:
                model.allparam.grad[:, :, 0].zero_()

            grad_norm = step(step_params, optimizer, clip_r=clip)

            if not np.isfinite(grad_norm):
                print(f'[DEBUG] [Norm]: infinite gradient norm at iteration {t}.')

            optimizer.zero_grad()

            # [P-FIX] Safeguard: re-zero P after the optimizer step.
            if freeze_p:
                model.zero_p()
            end = time.time()

            # Only prints for the first 5 iterations, and then every 100 iterations
            if t % 100 == 0 or t < 5:
                print(f'iter {t} | Loss: {loss.item()} | time: {end-start} | gradnorm: {grad_norm}')

        print(f'Finished seed={seed}, count={count}, type={model_type}, N={context_length}, '
              f'mode={data_mode}, heads={head_count}, attention={attention_type}.')

    if learn_u:
        torch.save({'hist_dict': hist_dict, 'u_hist_dict': u_hist_dict}, filename)
        print(f'Saved histories to {filename}')
        return hist_dict, u_hist_dict

    torch.save({'hist_dict': hist_dict}, filename)
    print(f'Saved histories to {filename}')
    return hist_dict


####################################
# Evaluation
####################################

def compute_loss_dict(hist_dict, U=None, D=None, test_seed=None,
                      data_mode: DataMode | None = None, head_counts: list[int] | None = None,
                      attention_types: list[str] | None = None, u_hist_dict=None):
    """Evaluate every checkpoint on a fixed test batch.

    Each run is evaluated with the attention type in its key, and by default on
    the data mode in its key. Pass `data_mode` to override the data mode for every
    run. `attention_types` (e.g. ['linear', 'softmax']) restricts evaluation to runs
    with those attention types; None evaluates every run;
    U and D transform covariates. `head_counts` is accepted for compatibility
    with multi-head sweeps. Pass the `u_hist_dict` returned by
    `run_experiment_suite(..., learn_u=True)` to evaluate runs with their learned u
    (runs missing from it are evaluated with u = 0).
    """
    if head_counts is None:
        head_counts = [n_head]
    else:
        head_counts = list(head_counts)

    loss_dict = {}
    for key, history in hist_dict.items():
        key_tuple = tuple(key)
        if len(key_tuple) not in {5, 6, 7}:
            raise ValueError(f'Unexpected key format in hist_dict: {key}')
        seed, count, model_type, context_length = key_tuple[:4]
        train_data_mode = key_tuple[4] if len(key_tuple) > 4 else normal_data_mode
        head_count = key_tuple[5] if len(key_tuple) > 5 else n_head
        run_attention_type = _attention_type(key_tuple)
        if attention_types is not None and run_attention_type not in attention_types:
            continue

        print(key)
        print(f"Testing model: seed={seed}, count={count}, type={model_type}, "
              f"N={context_length}, train data={train_data_mode}, heads={head_count}, "
              f"attention={run_attention_type}")
        run_mode = data_mode if data_mode is not None else train_data_mode
        test_losses = torch.full((len(range(0, max_iters, stride)),), float('nan'))

        np.random.seed(test_seed)
        torch.manual_seed(test_seed)

        Z, y = generate_data(run_mode, context_length, d, B, U=U, D=D)
        Z = Z.to(device)
        y = y.to(device)
        u_history = u_hist_dict.get(key) if u_hist_dict is not None else None
        model = build_transformer(
            model_type, count, head_count, d, var,
            attention_type=run_attention_type,
            learn_u=u_history is not None,
        ).to(device)

        for iteration in range(0, max_iters, stride):
            if iteration % hist_stride != 0:
                continue
            history_index = iteration // hist_stride
            if history_index >= len(history):
                break
            snapshot = history[history_index]
            if not torch.isfinite(snapshot).all():
                continue

            with torch.no_grad():
                model.allparam.copy_(snapshot.to(device))
                if u_history is not None:
                    model.u.copy_(u_history[history_index].to(device))
                test_losses[iteration // stride] = in_context_loss(model, Z, y).item()

        loss_dict[key] = test_losses

    return loss_dict


def compute_looped_loss_by_loop_count(hist_dict, max_loops=50, U=None, D=None, test_seed=None,
                                      data_mode: DataMode | None = None,
                                      attention_type: str | None = None):
    """Evaluate final looped checkpoints using 1..max_loops forward loops.

    ``hist_dict`` should contain looped-model histories only. Each result is a
    length-``max_loops`` tensor, where index ``k - 1`` contains the test loss
    using ``k`` loops. Attention type is read from each key when present, or
    can be explicitly overridden with ``attention_type``.
    """
    if test_seed is None:
        test_seed = TEST_SEED

    loss_dict = {}
    for key, history in hist_dict.items():
        print(f"Testing: {key}")
        seed, trained_loop_count, model_type, context_length, train_data_mode, head_count = key[:6]
        saved_attention_type = key[6] if len(key) > 6 else 'linear'
        run_attention_type = attention_type or saved_attention_type
        run_mode = data_mode if data_mode is not None else train_data_mode

        losses = torch.full((max_loops,), float('nan'))
        if not history or not torch.isfinite(history[-1]).all():
            loss_dict[key] = losses
            continue

        np.random.seed(test_seed)
        torch.manual_seed(test_seed)
        Z, y = generate_data(run_mode, context_length, d, B, U=U, D=D)
        Z = Z.to(device)
        y = y.to(device)

        # Reuse the normal model builder so linear and softmax evaluation share
        # the same checkpoint-loading and loop-count logic.
        model = build_transformer(
            'looped', 1, head_count, d, var, attention_type=run_attention_type,
        ).to(device)
        model.eval()
        with torch.no_grad():
            model.allparam.copy_(history[-1].to(device))
            for loop_count in range(1, max_loops + 1):
                model.n_loop = loop_count
                losses[loop_count - 1] = in_context_loss(model, Z, y).item()

        loss_dict[key] = losses
        print(f"Evaluated checkpoint {key} through {max_loops} loops.")

    return loss_dict


def compute_dist_identity(M):
    scale = torch.diagonal(M).sum() / M.shape[0]
    ideal_identity = scale * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    denominator = torch.linalg.norm(M, ord='fro')
    if not torch.isfinite(denominator) or denominator == 0:
        return float('nan')
    return (torch.linalg.norm(M - ideal_identity, ord='fro') / denominator).item()


def compute_dist_dict(hist_dict, iterations, n_head, hist_stride=1):
    """Return {key: tensor (count, P/Q, checkpoints)} of distance-to-identity.

    For looped models, the shared block is repeated across loop positions for a direct comparison.
    """
    dist_dict = {}

    for key, history in hist_dict.items():
        seed, count, model_type, context_length, data_mode = key[:5]
        run_head_count = key[5] if len(key) > 5 else n_head
        distances = torch.full((count, 2, len(iterations)), float('nan'))

        for point_index, iteration in enumerate(iterations):
            if iteration % hist_stride != 0:
                continue
            history_index = iteration // hist_stride
            if history_index >= len(history):
                break
            snapshot = history[history_index]
            if not torch.isfinite(snapshot).all():
                continue

            for block in range(count):
                source_block = block if model_type == 'layered' else 0
                for matrix_type in range(2):
                    head_distances = [
                        compute_dist_identity(snapshot[source_block, head, matrix_type])
                        for head in range(run_head_count)
                    ]
                    finite_distances = [value for value in head_distances if np.isfinite(value)]
                    if finite_distances:
                        distances[block, matrix_type, point_index] = float(np.mean(finite_distances))

        dist_dict[key] = distances

    return dist_dict


def compute_data_condition_numbers(Z):
    """Return per-task condition numbers of X and its uncentered Gram matrix X.T @ X.

    Z is the prompt returned by generate_data: the last row is the query and
    the last column contains labels. Singular Gram matrices have condition inf.
    """
    if Z.ndim != 3 or Z.shape[1] < 2 or Z.shape[2] < 2:
        raise ValueError('Expected Z with shape (B, N + 1, d + 1), N >= 1, d >= 1.')

    X = Z[:, :-1, :-1].double()
    singular_values = torch.linalg.svdvals(X)
    largest, smallest = singular_values[:, 0], singular_values[:, -1]
    tolerance = torch.finfo(X.dtype).eps * max(X.shape[-2:]) * largest
    condition_X = torch.where(smallest > tolerance, largest / smallest, float('inf'))
    if X.shape[-2] < X.shape[-1]:
        condition_gram = torch.full_like(condition_X, float('inf'))
    else:
        condition_gram = condition_X.square()
    return condition_X, condition_gram


####################################
# Plotting
####################################

def _attention_type(key):
    """Attention type stored in a run key (seventh field); linear runs omit it."""
    return key[6] if len(key) > 6 else 'linear'


def _data_mode_name(key):
    if len(key) > 4:
        data_mode = key[4]
        if isinstance(data_mode, DataMode):
            if data_mode.name == 'gamma':
                return f'gamma (shape={data_mode.gamma_shape}, scale={data_mode.gamma_scale})'
            return data_mode.name
        return data_mode
    return normal_data_mode.name


RUN_LABEL_FIELDS = ('heads', 'attention', 'distribution')


def _run_label_parts(key, labels):
    """Describe a run key with the requested fields, in the order given.

    `labels` may contain 'heads' (e.g. 'heads=2'), 'attention' (e.g. 'softmax attention')
    and 'distribution' (e.g. 'trained on normal'). Returns a list of strings.
    """
    parts = []
    for field in labels:
        if field == 'heads':
            parts.append(f'heads={key[5] if len(key) > 5 else n_head}')
        elif field == 'attention':
            parts.append(f'{_attention_type(key)} attention')
        elif field == 'distribution':
            parts.append(f'trained on {_data_mode_name(key)}')
        else:
            raise ValueError(f'Unknown label field {field!r}; expected one of {RUN_LABEL_FIELDS}')
    return parts


def _add_bottom_legend(fig, ax, fontsize=9, ncol=2, top=1):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        fig.tight_layout(rect=(0, 0, 1, top))
        return

    ncol = min(ncol, len(handles))
    nrows = (len(handles) + ncol - 1) // ncol
    fig.legend(
        handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
        fontsize=fontsize, ncol=ncol,
    )
    fig.tight_layout(rect=(0, 0.04 + 0.045 * nrows, 1, top))


# Distinct colors for loss curves; each curve in a figure takes the next one (cycling after 20).
CURVE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#000075', '#f58231', '#3cb44b', '#800000', '#911eb4',
    '#9a6324', '#f032e6', '#469990', '#808000', '#42d4f4',
]


def plot_loss_curves(loss_dict, n_layers_loops, model_types, iterations,
                     fig_dir='figures', filename='figure_two_test_loss_layered_vs_looped.pdf',
                     seeds=None, log_scale=True, figsize=(9, 6),
                     test_data_mode=None, head_count=None, head_counts=None, max_iters=None,
                     attention_types=None, labels=None):
    """Plot mean +/- std test loss over seeds for each (count, model_type), one figure per context length.

    `max_iters` truncates every curve to the points whose iteration is below `max_iters` (None plots all).

    `seeds=None` uses all seeds; each figure is saved as `filename` with `_N{context_length}` appended to the stem.
    When `test_data_mode` is set, training-mode curves share one figure for that test distribution.
    When `head_counts` is given, each head count (key[5]) gets its own curve and color on the same figure.
    When `attention_types` is given (e.g. ['linear', 'softmax']), each attention type gets its own
    labeled curve and color as well; combined with `head_counts`, every (heads, attention) pair does.

    `labels` lists what each curve's legend entry shows after the model name: any of
    'heads', 'attention' and 'distribution' (training distribution), in that order given.
    None shows the training distribution, plus heads/attention when `head_counts`/`attention_types` are set.
    """
    if labels is None:
        labels = (['heads'] if head_counts else []) + (['attention'] if attention_types else []) + ['distribution']
    stem, ext = os.path.splitext(filename)
    iterations = np.asarray(iterations)
    n_points = len(iterations) if max_iters is None else int((iterations < max_iters).sum())
    iterations = iterations[:n_points]

    data_modes = list(dict.fromkeys(_data_mode_name(key) for key in loss_dict))
    if test_data_mode is None:
        run_configurations = sorted({
            (key[3], _data_mode_name(key)) for key in loss_dict
        })
    else:
        run_configurations = [(context_length, None) for context_length in sorted({
            key[3] for key in loss_dict
        })]
    for context_length, data_mode in run_configurations:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        curve_index = 0  # every curve in this figure gets its own color

        for count in n_layers_loops:
            for model_type in model_types:
                line_style, make_label = MODEL_STYLES[model_type]
                training_modes = data_modes if test_data_mode is not None else [data_mode]
                curve_groups = list(product(head_counts or [None], attention_types or [None]))
                for training_mode, (curve_heads, curve_attention) in product(training_modes, curve_groups):
                    run_keys = [
                        key for key in loss_dict
                        if key[1] == count and key[2] == model_type and key[3] == context_length
                        and _data_mode_name(key) == training_mode
                        and (seeds is None or key[0] in seeds)
                        and (curve_heads is None or (len(key) > 5 and key[5] == curve_heads))
                        and (curve_attention is None or _attention_type(key) == curve_attention)
                    ]
                    if not run_keys:
                        continue
                    run_losses = torch.stack([loss_dict[key][:n_points] for key in run_keys])
                    valid = torch.isfinite(run_losses)
                    n_valid = valid.sum(dim=0)
                    safe_losses = torch.where(valid, run_losses, torch.zeros_like(run_losses))
                    mean_loss = safe_losses.sum(dim=0) / n_valid.clamp_min(1)
                    mean_loss[n_valid == 0] = float('nan')

                    variance = torch.where(
                        valid,
                        (run_losses - mean_loss.unsqueeze(0)) ** 2,
                        torch.zeros_like(run_losses),
                    ).sum(dim=0) / n_valid.clamp_min(1)
                    std_loss = variance.sqrt()
                    color = CURVE_COLORS[curve_index % len(CURVE_COLORS)]
                    curve_index += 1
                    label = ', '.join([make_label(count)] + _run_label_parts(run_keys[0], labels))
                    ax.plot(
                        iterations, mean_loss.numpy(), color=color, linestyle=line_style,
                        lw=2.2, label=label,
                    )
                    multiple = n_valid > 1
                    lower = torch.where(multiple, mean_loss - std_loss, torch.nan)
                    upper = torch.where(multiple, mean_loss + std_loss, torch.nan)
                    ax.fill_between(
                        iterations, lower.numpy(), upper.numpy(),
                        color=color, alpha=0.10,
                    )

        if test_data_mode is not None:
            test_data_label = _data_mode_name((None, None, None, None, test_data_mode))
            ax.set_title(f'N={context_length}, test data={test_data_label}')
        else:
            ax.set_title(f'N={context_length}, data={data_mode}' if len(data_modes) > 1 else f'N={context_length}')
        if head_count is not None:
            ax.set_title(f'{ax.get_title()}, heads={head_count}')
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('ICL Test Loss', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if log_scale:
            ax.set_yscale('log')
        _add_bottom_legend(fig, ax, fontsize=9, ncol=2)
        ax.grid(True, which='both', alpha=0.2)
        if test_data_mode is not None:
            mode_suffix = f'_test-{test_data_mode.name}'
            if test_data_mode.name == 'gamma':
                mode_suffix += f'-shape-{test_data_mode.gamma_shape}-scale-{test_data_mode.gamma_scale}'
        else:
            mode_suffix = f'_mode-{data_mode}' if len(data_modes) > 1 else ''
        head_suffix = f'_heads-{head_count}' if head_count is not None else ''
        fig.savefig(
            os.path.join(fig_dir, f'{stem}_N{context_length}{mode_suffix}{head_suffix}{ext}'),
            dpi=600,
        )
        plt.show()


def plot_parameter_matrices(hist_dict, plot_seed, n_layers_loops, model_types, n_head,
                            hist_stride=1, fig_dir='figures', head_count=None, labels=None,
                            matrices=None):
    """Plot layer/loop-0 P/Q matrices from each run's last finite checkpoint.

    `matrices` lists which matrices to plot ('B' for P, 'A' for Q); None plots both.
    `labels` lists which run fields the figure title shows ('heads', 'attention',
    'distribution'); None shows all three.

    Layered models have a separate block per layer; looped models share one block.
    """
    if labels is None:
        labels = ['heads', 'attention', 'distribution']
    if matrices is None:
        matrices = ['B', 'A']
    matrix_indices = {'B': 0, 'A': 1}
    unknown = [name for name in matrices if name not in matrix_indices]
    if unknown or not matrices:
        raise ValueError(f"matrices must be a non-empty list drawn from 'A' and 'B'; got {matrices}")
    plotted_head_count = n_head if head_count is None else head_count
    data_modes = list(dict.fromkeys(_data_mode_name(key) for key in hist_dict))
    attention_types = list(dict.fromkeys(_attention_type(key) for key in hist_dict))
    run_configurations = sorted({
        (key[3], _data_mode_name(key), _attention_type(key)) for key in hist_dict
    })
    matrix_suffix = '' if sorted(matrices) == ['A', 'B'] else f'_{"".join(matrices)}'
    for context_length, data_mode, attention_type in run_configurations:
        for count in n_layers_loops:
            for model_type in model_types:
                key = next((candidate for candidate in hist_dict
                            if candidate[:4] == (plot_seed, count, model_type, context_length)
                            and _data_mode_name(candidate) == data_mode
                            and _attention_type(candidate) == attention_type), None)
                if key is None:
                    print(f'No saved run found for {(plot_seed, count, model_type, context_length, data_mode, attention_type)}; skipping parameter plot.')
                    continue
                history = hist_dict[key]
                snapshot_index = next(
                    (idx for idx in range(len(history) - 1, -1, -1)
                     if torch.isfinite(history[idx]).all()),
                    None,
                )
                if snapshot_index is None:
                    print(f'No finite checkpoint found for {key}; skipping parameter plot.')
                    continue

                snapshot = history[snapshot_index].detach().cpu()
                fig, axs = plt.subplots(
                    len(matrices) * plotted_head_count, 1,
                    figsize=(4, 4 * len(matrices) * plotted_head_count),
                    squeeze=False,
                )

                for head in range(plotted_head_count):
                    for row_group, matrix_name in enumerate(matrices):
                        ax = axs[row_group * plotted_head_count + head, 0]
                        matrix = snapshot[0, head, matrix_indices[matrix_name]]
                        im = ax.imshow(matrix, cmap='gray_r')
                        for row in range(matrix.shape[0]):
                            for col in range(matrix.shape[1]):
                                ax.text(col, row, f'{matrix[row, col].item():.2f}',
                                        ha='center', va='center', color='r')
                        if model_type == 'looped':
                            title = f'shared {matrix_name}, loop 0, head {head}'
                        else:
                            title = f'{matrix_name}, layer 0, head {head}'
                        ax.set_title(title)
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                title_parts = [f'count={count}', f'N={context_length}', f'seed={plot_seed}']
                title_parts += _run_label_parts(key, labels)
                title_parts.append(f'checkpoint={snapshot_index * hist_stride}')
                fig.suptitle(f'{model_type}: {", ".join(title_parts)}', fontsize=14)
                fig.tight_layout(rect=(0, 0, 1, 0.93))
                mode_suffix = f'_{data_mode}' if len(data_modes) > 1 else ''
                attention_suffix = f'_attn-{attention_type}' if len(attention_types) > 1 else ''
                fig.savefig(
                    os.path.join(
                        fig_dir,
                        f'figure_two_parameters_{model_type}_count{count}_N{context_length}'
                        f'{mode_suffix}{attention_suffix}_heads-{plotted_head_count}{matrix_suffix}'
                        f'_seed{plot_seed}.pdf'),
                    dpi=600,
                    bbox_inches='tight',
                )
                plt.show()


def plot_dist_to_identity(dist_dict, n_layers_loops, iterations, fig_dir='figures', head_count=None,
                          labels=None):
    """Plot loop-0 mean +/- std distance-to-identity for Q (labeled A) only.

    Runs with different attention types (key[6]) share a figure, with one color per type.
    `labels` lists what each legend entry shows ('heads', 'attention', 'distribution');
    None shows attention and distribution.
    """
    if labels is None:
        labels = ['attention', 'distribution']
    data_modes = list(dict.fromkeys(_data_mode_name(key) for key in dist_dict))
    attention_types = list(dict.fromkeys(_attention_type(key) for key in dist_dict))
    attention_colors = {
        attention_type: color for attention_type, color in zip(
            attention_types, ['tab:orange', 'tab:blue', 'tab:green', 'tab:red'])
    }
    run_configurations = sorted({
        (key[3], _data_mode_name(key)) for key in dist_dict
    })
    for context_length, data_mode in run_configurations:
        for count in n_layers_loops:
            fig, ax = plt.subplots(figsize=(6, 4))

            matrix_type = 1  # Q parameter; shown as A in the identity plots.
            matrix_name = 'A'
            for (model_type, line_style, label), attention_type in product((
                ('layered', '-', 'baseline'),
                ('looped', '--', 'looping (shared weights)'),
            ), attention_types):
                run_keys = [
                    key for key in dist_dict
                    if key[1] == count and key[2] == model_type and key[3] == context_length
                    and _data_mode_name(key) == data_mode
                    and _attention_type(key) == attention_type
                ]
                if not run_keys:
                    continue

                seed_curves = torch.stack([
                    dist_dict[key][0, matrix_type] for key in run_keys
                ])
                valid = torch.isfinite(seed_curves)
                n_valid = valid.sum(dim=0)
                values = torch.where(valid, seed_curves, torch.zeros_like(seed_curves))
                mean = values.sum(dim=0) / n_valid.clamp_min(1)
                mean[n_valid == 0] = float('nan')
                variance = torch.where(
                    valid,
                    (seed_curves - mean.unsqueeze(0)) ** 2,
                    torch.zeros_like(seed_curves),
                ).sum(dim=0) / n_valid.clamp_min(1)
                std = variance.sqrt()

                color = attention_colors[attention_type]
                ax.plot(
                    iterations, mean.numpy(), color=color, linestyle=line_style,
                    lw=2, label=', '.join([f'{label} {matrix_name}'] + _run_label_parts(run_keys[0], labels)),
                )
                multiple = n_valid > 1
                lower = torch.where(multiple, mean - std, torch.nan)
                upper = torch.where(multiple, mean + std, torch.nan)
                ax.fill_between(
                    iterations, lower.numpy(), upper.numpy(),
                    color=color, alpha=0.10,
                )

            ax.set_title(f'{matrix_name}, layer/loop 0')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Distance to identity')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
            mode_title = f', data={data_mode}' if len(data_modes) > 1 else ''
            head_title = f', heads={head_count}' if head_count is not None else ''
            fig.suptitle(
                f'Distance to identity of A: count={count}, N={context_length}'
                f'{mode_title}{head_title}',
                fontsize=15,
            )
            _add_bottom_legend(fig, ax, fontsize=8, ncol=2, top=0.92)
            mode_suffix = f'_mode-{data_mode}' if len(data_modes) > 1 else ''
            head_suffix = f'_heads-{head_count}' if head_count is not None else ''
            fig.savefig(
                os.path.join(
                    fig_dir,
                    f'figure_two_dist_to_identity_A_count{count}_N{context_length}'
                    f'{mode_suffix}{head_suffix}.pdf',
                ),
                dpi=600,
                bbox_inches='tight',
            )
            plt.show()


def plot_loop_loss_curves(loop_loss_dict, n_layers_loops, loop_counts=None,
                         fig_dir='figures', filename='looped_test_loss_vs_loops.pdf',
                         seeds=None, log_scale=True, figsize=(9, 6)):
    """Plot ICL test loss versus inference loop count for looped-model checkpoints.

    `loop_loss_dict` maps each run key (seed, trained loop count, model type,
    context length) to a vector whose position k - 1 contains the loss using k
    loops. Pass looped-model runs only. Curves average over seeds for each
    trained loop count and context length.
    """
    stem, ext = os.path.splitext(filename)
    palette = plt.get_cmap('tab10')
    colors = {
        count: palette(index % 10)
        for index, count in enumerate(n_layers_loops)
    }

    context_lengths = sorted({key[3] for key in loop_loss_dict})
    for context_length in context_lengths:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for count in n_layers_loops:
            run_keys = [
                key for key in loop_loss_dict
                if key[1] == count and key[3] == context_length
                and (seeds is None or key[0] in seeds)
            ]
            if not run_keys:
                continue

            run_losses = torch.stack([loop_loss_dict[key].detach().cpu() for key in run_keys])
            valid = torch.isfinite(run_losses)
            n_valid = valid.sum(dim=0)
            safe_losses = torch.where(valid, run_losses, torch.zeros_like(run_losses))
            mean_loss = safe_losses.sum(dim=0) / n_valid.clamp_min(1)
            mean_loss[n_valid == 0] = float('nan')
            variance = torch.where(
                valid,
                (run_losses - mean_loss.unsqueeze(0)) ** 2,
                torch.zeros_like(run_losses),
            ).sum(dim=0) / n_valid.clamp_min(1)
            std_loss = variance.sqrt()

            x_values = np.arange(1, run_losses.shape[1] + 1)
            if loop_counts is not None:
                selected = [
                    loop_count for loop_count in loop_counts
                    if 1 <= loop_count <= run_losses.shape[1]
                ]
                indices = torch.tensor([loop_count - 1 for loop_count in selected])
                x_values = np.asarray(selected)
                mean_loss = mean_loss[indices]
                std_loss = std_loss[indices]
                n_valid = n_valid[indices]

            ax.plot(
                x_values, mean_loss.numpy(), color=colors[count], marker='o',
                lw=2.2, label=f'looping (l=1, loop={count})',
            )
            multiple = n_valid > 1
            lower = torch.where(multiple, mean_loss - std_loss, torch.nan)
            upper = torch.where(multiple, mean_loss + std_loss, torch.nan)
            ax.fill_between(
                x_values, lower.numpy(), upper.numpy(),
                color=colors[count], alpha=0.10,
            )

        ax.set_title(f'N={context_length}')
        ax.set_xlabel('Number of loops', fontsize=14)
        ax.set_ylabel('ICL Test Loss', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'{stem}_N{context_length}{ext}'), dpi=600)
        plt.show()
