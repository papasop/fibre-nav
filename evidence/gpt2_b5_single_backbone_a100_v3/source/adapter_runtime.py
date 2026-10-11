"""One backbone, explicit runtime adapter routing; not native weight-update internalisation."""
def activate(model,name,writable=False):
 model.set_adapter(name)
 # PEFT set_adapter may enable A as well: enforce the declared B-only chart.
 for n,p in model.named_parameters():
  p.requires_grad_(writable and ('lora_B.'+name+'.') in n)
 return [p for n,p in model.named_parameters() if ('lora_B.'+name+'.') in n]

def controller_parameters(model):
 return [p for n,p in model.named_parameters() if 'lora_B.controller.' in n]
