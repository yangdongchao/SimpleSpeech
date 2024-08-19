# Copyright (c) UniAudio teams
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# This implementation is inspired from
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py and
# https://github.com/clementchadebec/benchmark_VAE/blob/dfa0dcf6c79172df5d27769c09c860c42008baaa/src/pythae/models/vq_vae/vq_vae_utils.py#L81
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""
import typing as tp

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .distrib import broadcast_tensors, rank, is_distributed
# from tools.tokenizer.soundstream.utils.utils import is_primary
import logging
import random
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _find_tensors
import torch.optim
import torch.utils.data
from packaging import version
from omegaconf import OmegaConf


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None):
    # log_file_path is used when unit testing
    if is_logging_process():
        logging.config.dictConfig(
            OmegaConf.to_container(cfg.job_logging_config, resolve=True)
        )
        return logging.getLogger(name)


# from https://github.com/Lightning-AI/lightning-bolts/blob/5d61197cd2f491f69e238137a5edabe80ae14ad9/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20
class SyncFunction(torch.autograd.Function):
    @staticmethod
    # @torch.no_grad()
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode("utf-8")


class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):  # pragma: no cover
        if version.parse(torch.__version__[:6]) < version.parse("1.11"):
            self._sync_params()
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            assert len(self.device_ids) == 1
            if self.module.training:
                output = self.module.training_step(*inputs[0], **kwargs[0])
            elif self.module.testing:
                output = self.module.test_step(*inputs[0], **kwargs[0])
            else:
                output = self.module.validation_step(*inputs[0], **kwargs[0])
            if torch.is_grad_enabled():
                # We'll return the output object verbatim since it is a freeform
                # object. We need to find any tensors in this object, though,
                # because we need to figure out which parameters were used during
                # this forward pass, to ensure we short circuit reduction for any
                # unused parameters. Only if `find_unused_parameters` is set.
                if self.find_unused_parameters:
                    self.reducer.prepare_for_backward(list(_find_tensors(output)))
                else:
                    self.reducer.prepare_for_backward([])
        else:
            from torch.nn.parallel.distributed import \
                logging, Join, _DDPSink, _tree_flatten_with_rref, _tree_unflatten_with_rref
            with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.logger.set_runtime_stats_and_log()
                    self.num_iterations += 1
                    self.reducer.prepare_for_forward()

                # Notify the join context that this process has not joined, if
                # needed
                work = Join.notify_join_context(self)
                if work:
                    self.reducer._set_forward_pass_work_handle(
                        work, self._divide_by_initial_world_size
                    )

                # Calling _rebuild_buckets before forward compuation,
                # It may allocate new buckets before deallocating old buckets
                # inside _rebuild_buckets. To save peak memory usage,
                # call _rebuild_buckets before the peak memory usage increases
                # during forward computation.
                # This should be called only once during whole training period.
                if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
                    logging.info("Reducer buckets have been rebuilt in this iteration.")
                    self._has_rebuilt_buckets = True

                # sync params according to location (before/after forward) user
                # specified as part of hook, if hook was specified.
                buffer_hook_registered = hasattr(self, 'buffer_hook')
                if self._check_sync_bufs_pre_fwd():
                    self._sync_buffers()

                if self._join_config.enable:
                    # Notify joined ranks whether they should sync in backwards pass or not.
                    self._check_global_requires_backward_grad_sync(is_joined_rank=False)

                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])

                # sync params according to location (before/after forward) user
                # specified as part of hook, if hook was specified.
                if self._check_sync_bufs_post_fwd():
                    self._sync_buffers()

                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.require_forward_param_sync = True
                    # We'll return the output object verbatim since it is a freeform
                    # object. We need to find any tensors in this object, though,
                    # because we need to figure out which parameters were used during
                    # this forward pass, to ensure we short circuit reduction for any
                    # unused parameters. Only if `find_unused_parameters` is set.
                    if self.find_unused_parameters and not self.static_graph:
                        # Do not need to populate this for static graph.
                        self.reducer.prepare_for_backward(list(_find_tensors(output)))
                    else:
                        self.reducer.prepare_for_backward([])
                else:
                    self.require_forward_param_sync = False

            # TODO: DDPSink is currently enabled for unused parameter detection and
            # static graph training for first iteration.
            if (self.find_unused_parameters and not self.static_graph) or (
                    self.static_graph and self.num_iterations == 1
            ):
                state_dict = {
                    'static_graph': self.static_graph,
                    'num_iterations': self.num_iterations,
                }

                output_tensor_list, treespec, output_is_rref = _tree_flatten_with_rref(
                    output
                )
                output_placeholders = [None for _ in range(len(output_tensor_list))]
                # Do not touch tensors that have no grad_fn, which can cause issues
                # such as https://github.com/pytorch/pytorch/issues/60733
                for i, output in enumerate(output_tensor_list):
                    if torch.is_tensor(output) and output.grad_fn is None:
                        output_placeholders[i] = output

                # When find_unused_parameters=True, makes tensors which require grad
                # run through the DDPSink backward pass. When not all outputs are
                # used in loss, this makes those corresponding tensors receive
                # undefined gradient which the reducer then handles to ensure
                # param.grad field is not touched and we don't error out.
                passthrough_tensor_list = _DDPSink.apply(
                    self.reducer,
                    state_dict,
                    *output_tensor_list,
                )
                for i in range(len(output_placeholders)):
                    if output_placeholders[i] is None:
                        output_placeholders[i] = passthrough_tensor_list[i]

                # Reconstruct output data structure.
                output = _tree_unflatten_with_rref(
                    output_placeholders, treespec, output_is_rref
                )
        return output



def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10, frames_to_use: int = 10_000):
    """ Run K-means clustering on samples.
    Args:
        samples (tensor): shape [B * T, D]
        num_clusters (int): number of centroids.
        num_iters (int): number of iterations.
    """
    dim, dtype = samples.shape[-1], samples.dtype

    if frames_to_use < samples.shape[0]:
        samples = sample_vectors(samples, frames_to_use)

    # Init means
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Flag variable to indicate whether the codebook is initialized
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        # Runing EMA cluster size/count: N_i^t in eq. (6) in vqvae paper
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        # Codebook
        self.register_buffer("embed", embed)
        # EMA codebook: eq. (7) in vqvae paper
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        """ Initialize codebook.
        Args:
            data (tensor): [B * T, D].
        """
        if self.inited:
            return

        # if is_primary():
            # print(data.shape)
        
        ## NOTE (snippet added by Songxiang Liu): gather data from all gpus
        if dist.is_available() and dist.is_initialized():
            # [B * T * world_size, D]
            data = SyncFunction.apply(data)

        # if is_primary():
            # print(data.shape)

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        # if is_primary():
            # print(batch_samples.shape)

        ## NOTE (snippet added by Songxiang Liu): gather data from all gpus
        if is_distributed():
            # [B * T * world_size, D]
            batch_samples = SyncFunction.apply(batch_samples)

        # if is_primary():
            # print(batch_samples.shape)

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x) # [B, T, D] -> [B*T, D]
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        # shape: [B, T, D]
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x) # [B, T, D] -> [B*T, D]
        
        # Initialize codebook
        self.init_embed_(x)

        embed_ind = self.quantize(x) # [B*T,]
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype) # [B*T, cb-size]
        embed_ind = self.postprocess_emb(embed_ind, shape) # [B, T]
        quantize = self.dequantize(embed_ind) # [B, T, D]

        if self.training:
            ### Update codebook by EMA
            embed_onehot_sum = embed_onehot.sum(0) # [cb-size,]
            embed_sum = x.t() @ embed_onehot # [D, cb-size]
            # if is_primary():
                # print("--------- 1 ---------")
                # print(embed_onehot_sum[:10])
            if is_distributed():
                dist.all_reduce(embed_onehot_sum)
                dist.all_reduce(embed_sum)
            # if is_primary():
                # print(embed_onehot_sum[:10])
                # print("--------- 2 ---------")
            # Update ema cluster count N_i^t, eq. (6) in vqvae paper
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            # Update ema embed: eq. (7) in vqvae paper
            self.embed_avg.data.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)
            # apply laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
            )
            # Update ema embed: eq. (8) in vqvae paper
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
