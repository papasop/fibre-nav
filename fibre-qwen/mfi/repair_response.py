"""KL-aware direction plus local inward response correction, finite gates unchanged."""
import torch
from core import near_kernel
from repair_kl import KLAwareController

class ResponseKLController(KLAwareController):
    response_activation_fraction = 0.5
    response_reduction_fraction = 0.5

    def direction(self, value, jac, audit):
        d = near_kernel(-self.backend.gradient(value), jac, self.budget.ridge)
        norm = d.norm()
        if not torch.isfinite(norm) or norm <= 1e-12:
            return d
        d = d / norm * self.budget.step_norm
        if audit['response'] >= self.response_activation_fraction * self.budget.response:
            residual = (self.backend.response().detach().double() - self.backend.source_response)
            j = jac.double()
            correction = j.T @ torch.linalg.solve(j @ j.T + self.budget.ridge *
                torch.eye(len(j),dtype=j.dtype,device=j.device),
                -self.response_reduction_fraction * residual)
            d = d + correction.to(d.dtype)
        if audit['kl'] >= self.activation_fraction * self.budget.kl:
            normal = near_kernel(self.backend.kl_gradient(), jac, self.budget.ridge)
            n = normal.norm()
            if torch.isfinite(n) and n > 1e-12:
                normal = normal / n
                d = d - (torch.dot(d,normal).clamp_min(0) + self.inward_fraction*d.norm())*normal
        return d
