# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This is an integration test of the denoising algorithm. For a known data distribution that
is Gaussian, we sample using the analytically derived ground truth score and check
we retrieve correct moments of the data distribution.
"""


import pytest
import torch
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import dpm_solver, heun_denoiser, euler_maruyama_denoiser
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE, rotmat_to_rotvec


@pytest.mark.parametrize(
    "solver,denoiser_kwargs",
    [(dpm_solver, {}), (dpm_solver, {"noise": 0.5}), (heun_denoiser, {"noise": 0.5}), (euler_maruyama_denoiser, {"noise": 0.5})],
)
def test_reverse_sampling(solver, denoiser_kwargs):
    torch.manual_seed(1)
    N = 200  # Timesteps
    batch_size = 1000

    # Assume a ground truth distribution for 1D positions to be of mean -3 and std 4.3
    x0_mean = torch.tensor(-3.0)
    x0_std = torch.tensor(4.3)

    r3sde = CosineVPSDE()
    so3sde = DiGSO3SDE(num_sigma=10)
    sdes = {"pos": r3sde, "node_orientations": so3sde}

    def node_orientation_score(Rt: torch.Tensor, t: torch.Tensor):
        # Assume a ground truth distritubution for SO(3) is a delta distribution at the identity.
        # Note we use the gradient rotation vector for score instead of the matrix.
        grad_coefficients = so3sde.compute_score(rotmat_to_rotvec(Rt), t)
        assert grad_coefficients.shape == (batch_size, 3)
        return grad_coefficients

    def pos_score(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a_t, s_t = r3sde.marginal_prob(x=torch.ones_like(x_t), t=t)
        x0 = (x0_mean * s_t**2 + x_t * a_t * x0_std**2) / (s_t**2 + a_t**2 * x0_std**2)
        return (x0 * a_t - x_t) / s_t

    def score_fn(x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        expected_x0 = {
            "pos": pos_score(x.pos, t),
            "node_orientations": node_orientation_score(x.node_orientations, t),
        }
        return x.replace(**expected_x0)

    conditioning_data = Batch.from_data_list(
        [
            ChemGraph(
                pos=torch.randn(batch_size, 1),
                node_orientations=so3sde.prior_sampling((batch_size,)),
            )
        ]
    )

    samples = solver(
        sdes=sdes,
        batch=conditioning_data,
        N=N,
        score_model=score_fn,
        max_t=0.99,
        eps_t=0.001,
        device=torch.device("cpu"),
        **denoiser_kwargs,
    )

    assert torch.isclose(samples.pos.mean(), x0_mean, rtol=1e-1, atol=1e-1)
    assert torch.isclose(samples.pos.std().mean(), x0_std, rtol=1e-1, atol=1e-1)

    assert torch.allclose(samples.node_orientations.mean(dim=0), torch.eye(3), atol=1e-1)
    assert torch.allclose(samples.node_orientations.std(dim=0), torch.zeros(3, 3), atol=1e-1)

@pytest.mark.parametrize(
    "solver, N_per_batch, batch_size, N, denoiser_kwargs",
    [(euler_maruyama_denoiser, 100, 2, 200, {"noise": 0.0})],
)
def test_likelihood_computation(solver, N_per_batch, batch_size, N, denoiser_kwargs):
    """Check that the likelihood computation returns the correct value for a known distribution"""
    torch.manual_seed(1)
    # N = 200  # Timesteps
    # batch_size = 10
    # N_per_batch = 100


    # Assume a ground truth distribution for 1D positions to be of mean -3 and std 4.3
    x0_mean = torch.tensor(-3.0)
    # x0_std = torch.tensor(8.3)
    x0_std = torch.tensor(4.3)
    # x0_std = torch.tensor(1.0)

    gt_distribution = torch.distributions.Normal(loc=x0_mean, scale=x0_std)
    noise_distribution = torch.distributions.Normal(loc=0.0, scale=1.0)

    r3sde = CosineVPSDE()
    so3sde = DiGSO3SDE(num_sigma=10)
    sdes = {"pos": r3sde, "node_orientations": so3sde}

    def hutchinson_divergence_estimate(f, x, t, m):
        estimates = []
        x.requires_grad_(True)
        for _ in range(m):
            # Sample a random vector from Rademacher distribution
            v_last = v if 'v' in locals() else None
            
            v = torch.randint(0, 2, x.shape).float() * 2 - 1  # Shape (batch_size, dim)
            v = v.to(x.device)
            # Compute the divergence estimate using Hutchinson's estimator
            jvp_score = torch.autograd.functional.jvp(f, (x, t), (v, torch.zeros_like(t)))[1]  # Shape (batch_size, dim)

            div_f = torch.einsum("ni,ni->n", v, jvp_score)  # Shape (batch_size,n)
            estimates.append(div_f)
            
            
            # make sure were sampling different random vectors
            if v_last is not None:
                assert not torch.allclose(v, v_last), "Hutchinson's estimator is sampling the same random vector multiple times."
                
        mean_estimate = sum(estimates) / m
        return mean_estimate

    def node_orientation_score(Rt: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor) -> torch.Tensor:
        # Assume a ground truth distritubution for SO(3) is a delta distribution at the identity.
        # Note we use the gradient rotation vector for score instead of the matrix.
        grad_coefficients = so3sde.compute_score(rotmat_to_rotvec(Rt), t, batch_idx=batch_idx)
        # assert grad_coefficients.shape == Rt.shape, f"Expected shape of grad_coefficients to be {Rt.shape}, but got {grad_coefficients.shape}"
        return grad_coefficients

    def pos_score(x_t: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor) -> torch.Tensor:
        a_t, s_t = r3sde.marginal_prob(x=torch.ones_like(x_t), t=t, batch_idx=batch_idx)
        x0 = (x0_mean * s_t**2 + x_t * a_t * x0_std**2) / (s_t**2 + a_t**2 * x0_std**2)
        return (x0 * a_t - x_t) / s_t

    def score_fn(x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        pos = x.pos
        
        def div_fn(pos, t):
            return pos_score(pos, t, batch_idx=x.batch)
        
        expected_x0 = {
            "pos": pos_score(pos, t, batch_idx=x.batch),
            "node_orientations": node_orientation_score(x.node_orientations, t, batch_idx=x.batch),
            "div_pos_score": hutchinson_divergence_estimate(div_fn, pos, t, m=25)
        }
        return x.replace(**expected_x0)

    conditioning_data = Batch.from_data_list(
        [
            ChemGraph(
                pos=torch.randn(N_per_batch, 1),
                node_orientations=so3sde.prior_sampling((N_per_batch,)),
                likelihood =torch.zeros(N_per_batch),  # Placeholder for the divergence of the score function
                noise_likelihood = torch.zeros(N_per_batch), # Placeholder for the noise likelihood
            ) for _ in range(batch_size)
        ]
    )

    samples = euler_maruyama_denoiser(
        sdes=sdes,
        batch=conditioning_data,
        N=N,
        score_model=score_fn,
        max_t=0.999,
        eps_t=0.001,
        device=torch.device("cpu"),
        noise=0.0,
    )

    samples_batch = samples.to_data_list() 
    samples_pos = torch.stack([data.pos for data in samples_batch], dim=0).squeeze()  # Shape (batch_size,)
    samples_node_orientations = torch.stack([data.node_orientations for data in samples_batch], dim=0)  # Shape (batch_size, num_nodes, 3, 3)

    # compare likelihoods
    x0 = torch.stack([data.pos for data in samples_batch], dim=0).squeeze()  # Shape (batch_size,)
    x1 = torch.stack([data.pos for data in conditioning_data.to_data_list()], dim=0).squeeze()
    
    predicted_p0_likelihood = torch.stack([data.likelihood for data in samples_batch])
    true_p0_likelihood = gt_distribution.log_prob(x0)
    
    assert torch.allclose(predicted_p0_likelihood, true_p0_likelihood, atol=1e-1)
    
    

def test_dpm_solver_has_gradients(tiny_model, default_batch, sdes):
    """Check that DPM solver sampled coordinates have gradients w.r.t. model parameters."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples = dpm_solver(
        sdes=sdes,
        batch=default_batch,
        N=10,
        score_model=tiny_model,
        max_t=0.99,
        eps_t=0.001,
        device=device,
        record_grad_steps={1, 2, 3},
    )
    sum_pos = samples.pos.sum()

    params = [p for p in tiny_model.parameters() if p.requires_grad]
    assert len(params) > 0
    assert all([x.grad is None for x in params])
    sum_pos.backward()
    assert not all([torch.all(x.grad == 0) for x in params])
