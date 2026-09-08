"""Transactional L1 operations on a restricted neural parameter chart."""
from dataclasses import dataclass, asdict
import math
import torch


@dataclass(frozen=True)
class Budget:
    response: float = 0.02
    kl: float = 0.01
    margin: float = 1.0
    steps: int = 24
    step_norm: float = 0.5
    backtracks: int = 8
    ridge: float = 1e-8

    def __post_init__(self):
        values = [self.response, self.kl, self.margin, self.step_norm, self.ridge]
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError('Budgets and step parameters must be finite and positive')
        if not 1 <= self.steps <= 100 or not 1 <= self.backtracks <= 16:
            raise ValueError('Invalid iteration limits')


def near_kernel(direction, jac, ridge):
    j = jac.double(); v = direction.double()
    gram = j @ j.T
    return (v - j.T @ torch.linalg.solve(
        gram + ridge * torch.eye(len(j), dtype=j.dtype, device=j.device), j @ v)).to(direction.dtype)


class MFIController:
    """Externally commanded controller; never interprets arbitrary model instructions."""
    def __init__(self, backend, budget=Budget()):
        self.backend, self.budget = backend, budget
        self.written = False

    def read(self):
        return self.backend.read()

    def execute(self, operation, value):
        if operation not in ('WRITE', 'OVERWRITE') or type(value) is not int or value not in (0, 1):
            raise ValueError('Only WRITE/OVERWRITE with binary integer values are supported')
        if operation == 'OVERWRITE' and not self.written:
            raise ValueError('WRITE must succeed before OVERWRITE')
        if operation == 'WRITE' and self.written:
            raise ValueError('Use OVERWRITE for an initialized cell')
        b = self.backend; cfg = self.budget
        initial = b.vector().clone()
        before = b.read(); records = []; accepted = False
        try:
            for step in range(cfg.steps):
                audit = b.audit(value)
                if self.passes(audit):
                    break
                current = b.vector().clone()
                jac = b.jacobian()
                direction = near_kernel(-b.gradient(value), jac, cfg.ridge)
                norm = direction.norm()
                if not torch.isfinite(norm) or norm <= 1e-12:
                    break
                direction = direction / norm * cfg.step_norm
                residual = float((jac.double() @ direction.double()).norm())
                moved = False
                for k in range(cfg.backtracks):
                    b.set_vector(current + direction * (0.5 ** k))
                    candidate = b.audit(value)
                    if self.in_budget(candidate) and candidate['margin'] > audit['margin']:
                        moved = True
                        break
                if not moved:
                    b.set_vector(current)
                records.append({'step': step + 1, 'accepted_step': moved,
                                'projection_residual': residual,
                                'audit': b.audit(value)})
                print(operation, step+1, records[-1]['audit'], flush=True)
                if not moved:
                    break
            final = b.audit(value)
            proposal_read = b.read()
            accepted = self.passes(final)
        finally:
            # A proposal is never retained on exception or failed finite gates.
            if not accepted:
                b.set_vector(initial)
        if accepted:
            self.written = True
        return {'operation': operation, 'target': value, 'accepted': accepted,
                'before': before, 'proposal_read': proposal_read,
                'committed_read': b.read(), 'proposal_audit': final,
                'rolled_back': not accepted, 'trace': records,
                'budget': asdict(cfg)}

    def in_budget(self, audit):
        return all(math.isfinite(audit[k]) for k in ('response', 'kl', 'margin')) and (
            0 <= audit['response'] <= self.budget.response and
            -1e-10 <= audit['kl'] <= self.budget.kl)

    def passes(self, audit):
        return self.in_budget(audit) and audit['margin'] >= self.budget.margin
