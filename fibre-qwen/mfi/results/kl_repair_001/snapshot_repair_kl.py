"""Candidate KL-aware writer; original finite gates and rollback retained."""
import torch
from dataclasses import asdict
from core import MFIController, near_kernel
from qwen_l1 import QwenChart, ANCHORS


class QwenKLChart(QwenChart):
    def kl_gradient(self):
        total = torch.zeros_like(self.vector())
        for text, lp0 in zip(ANCHORS, self.anchor_logp):
            lp = self.logits(text).double().log_softmax(-1)
            loss = (lp0.exp() * (lp0 - lp)).sum() / len(ANCHORS)
            grads = torch.autograd.grad(loss, self.ps)
            total += torch.cat([g.detach().flatten() for g in grads])
        return total


class KLAwareController(MFIController):
    activation_fraction = 0.5
    inward_fraction = 0.25

    def direction(self, value, jac, audit):
        direction = near_kernel(-self.backend.gradient(value), jac, self.budget.ridge)
        if audit['kl'] < self.activation_fraction * self.budget.kl:
            return direction
        normal = near_kernel(self.backend.kl_gradient(), jac, self.budget.ridge)
        norm = normal.norm()
        if not torch.isfinite(norm) or norm <= 1e-12:
            return direction
        normal = normal / norm
        # Keep existing inward motion; remove outward motion and add a small
        # inward component to leave room for finite-step curvature.
        outward = torch.dot(direction, normal).clamp_min(0)
        return direction - (outward + self.inward_fraction * direction.norm()) * normal

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
                direction = self.direction(value, jac, audit)
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

