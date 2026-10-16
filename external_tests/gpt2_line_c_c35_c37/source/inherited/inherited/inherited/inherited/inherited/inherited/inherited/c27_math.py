"""Numerical fitting in saved training geometry, not a feasibility certificate."""
import numpy as np
import torch
from c27_protocol import require,SETTINGS


def residual_candidates(J,Phi,E):
 from joint_calibration import prepare_system
 from budget_calibration import interval_coefficients,reconstruct_direction,scale_direction
 system=prepare_system(J,Phi,E,SETTINGS['output_scale'],1e-12,1e-8);bases=[];infos=[]
 for ridge in SETTINGS['ridge_grid']:
  c,info=interval_coefficients(system,ridge,SETTINGS['target_response_interval'],SETTINGS['interval_solver_max_iterations'],SETTINGS['interval_solver_tolerance'],SETTINGS['interval_solver_polish_iterations'])
  bases.append(reconstruct_direction(system,c));infos.append(info)
 return system,bases,infos


def trial_matrix(system,bases,spec):
 from radius_calibration import scale_direction
 direction,info=scale_direction(system,bases[spec['ridge_index']],spec['ridge'],SETTINGS['layer_radius'],spec['increment_radius'])
 m=torch.from_numpy(spec['alpha']*direction).float()
 require(torch.isfinite(m).all(),'Nonfinite residual candidate')
 return m,info
