"""Model-output wiring only. This module neither imports nor computes task truth."""
import json
from interface_compiler import parse_input,compile_content

def decoded_bit(observation):
    if observation.get('accepted') is not True:return None
    c=observation.get('constrained',{})
    raw=c.get('raw')
    if raw not in ('0','1') or c.get('truncated') is not False:return None
    if c.get('output_ids')!=[15+int(raw),151645]:return None
    return int(raw)

def resolve_fields(item,previous):
    if item['kind']=='direct':return dict(item['fields']),None
    q=item['program']
    if item['kind']=='step1':return dict(op=q['op1'],a=q['a'],b=q['b']),None
    first=previous[item['first_index']]
    if first['item']['id']!=q['id']+'-step1':raise ValueError('Wrong dependency')
    bit=decoded_bit(first['observation'])
    link=dict(first_index=item['first_index'],first_id=first['item']['id'],observed_bit=bit,
              intervention='flip' if item['kind']=='flip_step2' else 'none')
    if bit is None:return None,link
    return dict(op=q['op2'],a=1-bit if item['kind']=='flip_step2' else bit,b=q['c']),link

def execute_item(item,previous,evaluator):
    fields,link=resolve_fields(item,previous)
    if fields is None:
        return dict(item=item,fields=None,link=link,source_text=None,
                    observation=dict(accepted=False,model_invocations=0,skip_reason='UPSTREAM_INVALID_BIT'))
    source=json.dumps(fields,separators=(',',':'))
    # Parse here to fail early; real process_input independently parses before inference.
    if parse_input(source)!=fields:raise ValueError('Serialization error')
    return dict(item=item,fields=fields,link=link,source_text=source,observation=evaluator(source))

def validate_record(record,item,previous,prompts):
    if record['item']!=item:raise ValueError('Schedule or item changed')
    fields,link=resolve_fields(item,previous)
    if record['fields']!=fields or record['link']!=link:raise ValueError('Dependency record changed')
    obs=record['observation']
    if fields is None:
        if record['source_text'] is not None or obs!=dict(accepted=False,model_invocations=0,skip_reason='UPSTREAM_INVALID_BIT'):
            raise ValueError('Invalid skipped record')
        return False
    if record['source_text']!=json.dumps(fields,separators=(',',':')):raise ValueError('Input changed')
    if obs.get('accepted') is not True:return False
    return (obs.get('model_invocations')==1 and obs.get('parse_error') is None
            and obs.get('parsed')==fields and obs.get('compiled_content')==compile_content(fields)
            and obs.get('compiled_prompt')==prompts[(fields['op'],fields['a'],fields['b'])])
