"""Declared grammar -> typed fields -> canonical prompt. No Boolean solver or model metadata.
Only these accepted forms are supported; this is an external deterministic compiler.
"""
import json,re
RULES={
 'OR':'For OR, output 1 when at least one input bit is 1, otherwise output 0.',
 'AND':'For AND, output 1 when both input bits are 1, otherwise output 0.',
 'XOR':'For XOR, output 1 when the two input bits have different values, otherwise output 0.',
 'EQUAL':'For A == B, output 1 when the two input bits have the same value, otherwise output 0.'}
SUFFIX='Return the Boolean result as one digit (0 or 1).'
OP_LABEL={'OR':'OR','AND':'AND','XOR':'XOR','EQUAL':'A == B'}
LABEL_OP={v:k for k,v in OP_LABEL.items()}
class ParseError(ValueError):pass

def validate_fields(fields):
    if not isinstance(fields,dict) or set(fields)!={'op','a','b'}:raise ParseError('FIELDS_REQUIRED_EXACTLY_OP_A_B')
    if not isinstance(fields['op'],str) or fields['op'] not in RULES:raise ParseError('UNSUPPORTED_OPERATION')
    if any(type(fields[k]) is not int or fields[k] not in (0,1) for k in ('a','b')):raise ParseError('INTEGER_BITS_REQUIRED')
    return dict(op=fields['op'],a=fields['a'],b=fields['b'])

def unique_object(pairs):
    obj={}
    for k,v in pairs:
        if k in obj:raise ParseError('DUPLICATE_JSON_FIELD')
        obj[k]=v
    return obj

def parse_input(text):
    if not isinstance(text,str) or not text or len(text)>4096:raise ParseError('INPUT_TEXT_REQUIRED_MAX4096')
    text=text.strip()
    if text.startswith('{'):
        try:obj=json.loads(text,object_pairs_hook=unique_object)
        except (ValueError,TypeError,RecursionError) as exc:raise ParseError('INVALID_JSON_OR_DUPLICATE_FIELD') from exc
        return validate_fields(obj)
    supplied_rule=None
    first,sep,rest=text.partition('\n')
    if first in RULES.values():supplied_rule=first;text=rest
    label=r'(?P<op>OR|AND|XOR|A == B)';bit_a=r'(?P<a>[01])';bit_b=r'(?P<b>[01])'
    suffix=re.escape(SUFFIX)
    patterns=[r'A = '+bit_a+r'\nB = '+bit_b+r'\nOperation: '+label+r'\n'+suffix,
              r'The input bits are A: '+bit_a+r'; B: '+bit_b+r'\.\nApply '+label+r' to these two bits\.\n'+suffix,
              r'Evaluate '+label+r' on this record:\nB = '+bit_b+r'\nA = '+bit_a+r'\n'+suffix]
    matches=[m for pattern in patterns if (m:=re.fullmatch(pattern,text))]
    if len(matches)!=1:raise ParseError('UNSUPPORTED_OR_AMBIGUOUS_SYNTAX')
    m=matches[0];fields=validate_fields(dict(op=LABEL_OP[m['op']],a=int(m['a']),b=int(m['b'])))
    if supplied_rule is not None and supplied_rule!=RULES[fields['op']]:raise ParseError('RULE_OPERATION_CONFLICT')
    return fields

def compile_content(fields):
    f=validate_fields(fields)
    return RULES[f['op']]+'\n'+f"A = {f['a']}\nB = {f['b']}\nOperation: {OP_LABEL[f['op']]}\n"+SUFFIX

def compile_prompt(tok,fields):
    content=compile_content(fields)
    prompt=tok.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer: '
    return content,prompt

def process_input(text,tok,infer):
    # No test label, expected fields, case identifier or truth function is passed here.
    try:parsed=parse_input(text)
    except ParseError as exc:return dict(accepted=False,parse_error=str(exc),parsed=None,compiled_content=None,compiled_prompt=None,model_invocations=0)
    content,prompt=compile_prompt(tok,parsed)
    result=infer(prompt)
    return dict(accepted=True,parse_error=None,parsed=parsed,compiled_content=content,compiled_prompt=prompt,model_invocations=1,**result)
