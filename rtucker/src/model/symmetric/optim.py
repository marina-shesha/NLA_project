from typing import Callable, Union

import torch

from torch.optim import Optimizer
from tucker_riemopt import SFTucker
from tucker_riemopt import SFTuckerRiemannian
from tucker_riemopt.sf_tucker.riemannian import TangentVector


class RGD(Optimizer):
    def __init__(self, params, rank, max_lr):
        self.rank = rank
        self.max_lr = max_lr
        self.lr = max_lr

        defaults = dict(rank=rank, max_lr=self.max_lr, lr=self.lr)
        super().__init__(params, defaults)

        self.direction = None
        self.loss = None

    def fit(self, loss_fn: Callable[[SFTucker], float], x_k: SFTuckerRiemannian,
            normalize_grad: Union[float, "False"] = 1.):
        """
                Computes the Riemannian gradient of `loss_fn` at point `x_k`.

                :param loss_fn: smooth scalar-valued loss function
                :param x_k: current solution approximation
                :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
                 be normalized to `normalize_grad`.
                :return: Frobenius norm of the Riemannian gradient.
                """
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()
        normalize_grad = rgrad_norm if not normalize_grad else normalize_grad
        normalizer = 1 / rgrad_norm * normalize_grad

        self.direction = normalizer * rgrad
        return rgrad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + SFTuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)


def f_tangent_vec(vec: TangentVector, f) -> TangentVector:
    return TangentVector(
        vec.point,
        f(vec.delta_core),
        [f(factor) for factor in vec.delta_regular_factors],
        f(vec.delta_shared_factor),
    )

def sq_vec(vec: TangentVector) -> TangentVector:
    return f_tangent_vec(vec, lambda x : torch.square(x))

def root_vec(vec: TangentVector) -> TangentVector:
    return f_tangent_vec(vec, lambda x : torch.sqrt(x))

def bin_f_vec(lhs: TangentVector, rhs: TangentVector, f) -> TangentVector:
    return TangentVector(
        lhs.point,
        f(lhs.delta_core, rhs.delta_core),
        [
            f(fact_lhs,fact_rhs) for (fact_lhs, fact_rhs) in zip(
                lhs.delta_regular_factors, rhs.delta_regular_factors)
        ],
        f(lhs.delta_shared_factor, rhs.delta_shared_factor)
    )

def vec_add(lhs: TangentVector, rhs: float) -> TangentVector:
    return bin_f_vec(lhs, TangentVector(lhs.point, torch.ones_like(lhs.delta_core),
        [torch.full_like(factor, rhs) for factor in lhs.delta_regular_factors],
        torch.full_like(lhs.delta_shared_factor, rhs),
        ),
        lambda x, y: x + y
    )

def vec_div(lhs: TangentVector, rhs: TangentVector) -> TangentVector:
    return bin_f_vec(lhs, rhs, lambda x, y : x / y)

def vec_mul(lhs: TangentVector, rhs: TangentVector) -> TangentVector:
    return bin_f_vec(lhs, rhs, lambda x, y : x * y)

class AdaDelta(RGD):
    def __init__(self, params, rank, max_lr, betas=(0.1, 0.1), eps=1e-8):
        super().__init__(params, rank, max_lr)
        self.betas = betas
        self.eps = eps

        self.momentum = None
        self.second_momentum = None

    def fit(self, loss_fn: Callable[[SFTucker], float], x_k: SFTucker,
            normalize_grad: Union[float, "False"] = 1.):
        """
                Computes the Riemannian gradient of `loss_fn` at point `x_k`.

                :param loss_fn: smooth scalar-valued loss function
                :param x_k: current solution approximation
                :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
                 be normalized to `normalize_grad`.
                :return: Frobenius norm of the Riemannian gradient.
                """
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
                # retain_graph=True)
        # rgrad_norm = rgrad.norm().detach()

        if self.second_momentum is not None:
            self.second_momentum = (
                self.betas[1] * self.second_momentum +
                (1 - self.betas[1]) * sq_vec(rgrad)
            )
        else:
            self.second_momentum = (
                (1 - self.betas[1]) * sq_vec(rgrad)
            )

        if self.momentum is not None:
            self.momentum = TangentVector(x_k, torch.zeros_like(x_k.core))

        self.direction = SFTuckerRiemannian.project(
            x_k,
            vec_mul(
                vec_div(
                    root_vec(
                        vec_add(self.momentum, self.eps)
                    ),
                    root_vec(
                        vec_add(self.second_momentum, self.eps)
                    )
                ),
                rgrad
            ).construct(),
            # retain_graph=True
        )

        self.momentum = (
            self.betas[0] * self.momentum +
            (1 - self.betas[0]) * sq_vec(self.direction)
        )

        return self.direction.norm().detach()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + SFTuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)


class LRGeomCG(RGD):
    def __init__(self, params, rank, max_lr, betas=(0.1, 0.1), eps=1e-8):
        super().__init__(params, rank, max_lr)
        self.betas = betas
        self.eps = eps

        self.xi = None
        self.eta = None
        self.t = 0
        self.prev_xi = None

    def fit(self, loss_fn: Callable[[SFTucker], float], x_k: SFTucker,
            normalize_grad: Union[float, "False"] = 1.):
        """
                Computes the Riemannian gradient of `loss_fn` at point `x_k`.

                :param loss_fn: smooth scalar-valued loss function
                :param x_k: current solution approximation
                :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
                 be normalized to `normalize_grad`.
                :return: Frobenius norm of the Riemannian gradient.
                """
        self.xi, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k,
                retain_graph=True)

        if self.prev_xi is None:
            self.prev_xi = self.xi

        if self.eta is not None:
            prev_xi_constructed = self.prev_xi.construct()
            xi_transported = SFTuckerRiemannian.project(x_k, prev_xi_constructed,
                    retain_graph=True)
            eta_transported = SFTuckerRiemannian.project(x_k,
                    self.eta.construct(), retain_graph=True)

            delta = self.xi + (-xi_transported)
            beta = max(0, delta.construct().flat_inner(self.xi.construct()) /
                prev_xi_constructed.flat_inner(prev_xi_constructed))

            self.eta = -self.xi + beta * eta_transported
        else:
            self.eta = -self.xi

        eta_tensor = self.eta.construct()
        self.t = -eta_tensor.flat_inner(x_k) / eta_tensor.flat_inner(eta_tensor)

        self.prev_xi = self.xi

        return self.t * self.eta.norm().detach()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        assert self.xi

        W, E, R = self.param_groups[0]["params"]

        x_k = self.xi.point
        x_k = (
            self.param_groups[0]["lr"] * 0.5 * self.t * self.eta +
            SFTuckerRiemannian.TangentVector(x_k)
        )
        x_k = x_k.construct().round(self.rank)


        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)


class RSGDwithMomentum(RGD):
    def __init__(self, params, rank, max_lr, momentum_beta=0.9):
        super().__init__(params, rank, max_lr)
        self.momentum_beta = momentum_beta
        self.momentum = None

    def fit(self, loss_fn: Callable[[SFTucker], float], x_k: SFTucker,
            normalize_grad: Union[float, "False"] = 1.):
        """
        Computes the Riemannian gradient of `loss_fn` at point `x_k`.

        :param loss_fn: smooth scalar-valued loss function
        :param x_k: current solution approximation
        :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
         be normalized to `normalize_grad`.
        :return: Frobenius norm of the Riemannian gradient.
        """
        if self.direction is not None:
            self.momentum = SFTuckerRiemannian.project(x_k, self.direction)
        else:
            self.momentum = SFTuckerRiemannian.TangentVector(x_k,  torch.zeros_like(x_k.core))
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()
        normalize_grad = rgrad_norm if not normalize_grad else normalize_grad
        self.direction = (1 / rgrad_norm * normalize_grad) * rgrad + self.momentum_beta * self.momentum
        return rgrad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + SFTuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)
        self.direction = self.direction.construct()

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)


class SFTuckerAdam(RGD):
    def __init__(self, params, rank, max_lr, betas=(0.9, 0.999), eps=1e-8, step_velocity=1):
        super().__init__(params, rank, max_lr)
        self.betas = betas
        self.eps = eps
        self.step_velocity = step_velocity

        self.momentum = None
        self.second_momentum = torch.zeros(1, device="cuda")

        self.step_t = 1

    def fit(self, loss_fn: Callable[[SFTucker], float], x_k: SFTucker,
            normalize_grad: Union[float, "False"] = 1.):
        """
        Computes the Riemannian gradient of `loss_fn` at point `x_k`.

        :param loss_fn: smooth scalar-valued loss function
        :param x_k: current solution approximation
        :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
         be normalized to `normalize_grad`.
        :return: Frobenius norm of the Riemannian gradient.
        """
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()
        if self.momentum is not None:
            self.momentum = SFTuckerRiemannian.project(x_k, self.momentum.construct())
            self.momentum = self.betas[0] * self.momentum + (1 - self.betas[0]) * rgrad
        else:
            self.momentum = (1 - self.betas[0]) * rgrad
        self.second_momentum = self.betas[1] * self.second_momentum + (1 - self.betas[1]) * rgrad_norm ** 2
        second_momentum_corrected = self.second_momentum / (1 - self.betas[1] ** (self.step_t // self.step_velocity + 1))
        bias_correction_ratio = (1 - self.betas[0] ** (self.step_t // self.step_velocity + 1)) * torch.sqrt(
            second_momentum_corrected
        ) + self.eps
        self.direction = (1 / bias_correction_ratio) * self.momentum
        return rgrad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + SFTuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)

        self.step_t += 1
