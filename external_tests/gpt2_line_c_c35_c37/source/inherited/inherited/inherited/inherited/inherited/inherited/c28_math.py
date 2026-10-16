import numpy as np,torch
from c28_protocol import SETTINGS,require

def directions(J,H,E):
 from joint_calibration import prepare_system
 from budget_calibration import interval_coefficients,reconstruct_direction
 s=prepare_system(J,H,E,SETTINGS['output_scale'],1e-12,1e-8);bases=[];infos=[]
 for ridge in SETTINGS['ridge_grid']:
  c,info=interval_coefficients(s,ridge,SETTINGS['target_response_interval'],SETTINGS['interval_solver_max_iterations'],SETTINGS['interval_solver_tolerance'],SETTINGS['interval_solver_polish_iterations']);bases.append(reconstruct_direction(s,c));infos.append(info)
 return s,bases,infos

def candidate(system,bases,spec):
 from radius_calibration import scale_direction
 m,info=scale_direction(system,bases[spec['ridge_index']],spec['ridge'],SETTINGS['layer_radius'],spec['increment_radius']);m=torch.from_numpy(m*spec['alpha']).float();require(bool(torch.isfinite(m).all()),'Nonfinite candidate');return m,info
