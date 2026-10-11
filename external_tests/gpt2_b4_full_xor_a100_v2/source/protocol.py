PROTOCOL='GPT2_B4_FULL_XOR_A100_V2'
SEEDS=[82931,82961,82991]
DEVELOPMENT_SEED=82931
CASES=[(0,0),(0,1),(1,0),(1,1)]
INITIAL=[0,0,1,1]
SUBSTITUTED=[1,0,1,1]
FRESH_PREFIXES=[
 'For the sealed read test, reveal the symbol retained at',
 'Inspection of the persistent register requests the payload of',
 'The verification console asks which symbol occupies',
 'Retrieve the registered content identified by',
]
def fresh_prompts(cells):
 return [f'{prefix} [address={address}]:' for prefix in FRESH_PREFIXES for address in cells]
def controller(reader, blocked=False, replacement=None):
 if blocked:
  if replacement is None:raise ValueError('Blocked read requires declared replacement bits')
  a,b=replacement
 else:a,b=reader(0),reader(1)
 if a not in (0,1) or b not in (0,1):raise ValueError('Invalid decoded bit')
 return {'read_bits':[a,b],'operation':'XOR','destination':3,'value':a^b,
         'executor':'external deterministic XOR controller','read_blocked':blocked,
         'replacement_bits':list(replacement) if blocked else None}
def endpoint_pass(ev,drift,kl,cfg):
 return (ev['exact_accuracy']==ev['heldout_accuracy']==1
  and ev['minimum_exact_margin']>=.05 and ev['minimum_heldout_margin']>=.05
  and drift<=cfg['response_budget'] and kl<=cfg['endpoint_kl_gate'])
def case_checks(a,b,stages,traces,reset_bits,reset_budget_ok,reloads):
 expected=a^b
 prepare=stages['prepare']['observed_bits'];normal=stages['normal_writeback']['observed_bits']
 blocked=stages['blocked_writeback']['observed_bits'];sub=stages['input_substitution']['observed_bits']
 final=stages['substituted_writeback']['observed_bits']
 return {
  'all_development_endpoint_gates':all(s['pass'] for s in stages.values()),
  'initial_state':prepare==[a,b,1,1^expected],
  'actual_input_reads':traces['normal']['read_bits']==[a,b],
  'normal_xor_task':normal==[a,b,1,expected],
  'normal_required_output_flip':prepare[3]!=normal[3],
  'normal_unselected_retention':prepare[:3]==normal[:3],
  'executed_blocked_read_wrong_result':traces['blocked']['read_blocked'] and blocked[:3]==normal[:3] and blocked[3]==(1^expected),
  'blocked_write_required_flip':blocked[3]!=normal[3],
  'input_substitution_only_a':sub==[1^a,b,1,expected],
  'substitution_changes_computed_value':traces['substituted']['value']!=traces['normal']['value'],
  'substituted_xor_task':final==[1^a,b,1,1^expected],
  'substituted_required_output_flip':sub[3]!=final[3],
  'substituted_unselected_retention':sub[:3]==final[:3],
  'no_write_counterfactual_wrong':prepare[3]!=expected and sub[3]!=(1^expected),
  'baseline_reset_budget':reset_budget_ok,
  'reset_changes_stored_contents':reset_bits!=normal,
  'endpoint_reload':all(reloads.values()),
 }
