import zipfile
from pathlib import Path
from c31_protocol import *


def main():
    if not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    ps=list(Path('/content/gpt2_c31_residual_coverage_a100_v1').glob('session_*/'+RESULT+'.zip'))
    ps+=list(Path('/content/drive/MyDrive/'+DRIVE_NAME).glob(RESULT+'_*.zip'))
    good=[]
    for p in sorted(ps,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        try:
            with zipfile.ZipFile(p) as z:
                verify(z,True);require(zread(z,'identity.json')['protocol']==PROTOCOL,'Wrong protocol')
                s=zread(z,'SUMMARY.json') if 'SUMMARY.json' in z.namelist() else None
                good.append((s is not None,p,s))
        except Exception as e:print('跳过不可用包',p.name,str(e))
    require(bool(good),'没有找到可读C31结果')
    complete=[x for x in good if x[0]];_,p,s=(complete or good)[0]
    print('文件：',p,'\n大小：',p.stat().st_size,'bytes\nZIP SHA256：',sha(p),'\nCRC及成员哈希：通过')
    if s is None:print('尚无最终SUMMARY，请继续恢复本版C31。');return
    compact={k:s[k] for k in ('decision','complete','historical_training','new_training','prospective_local_development')}
    compact['prospective_prediction_comparison']={a:{f:r for f,r in rs.items() if f in ('route_mean','train_selected')} for a,rs in s['prospective_prediction_comparison'].items()}
    for k in ('frozen_predictor_sha256','C27_writer_unchanged','new_writer_training_steps','post_write_correction_steps','teacher_fallback_used','all48_training_and16_prospective_states_retained','generation_retries','offline_geometry_saved','fresh_evaluation','internal_parameter_update','LC1_closed'):compact[k]=s[k]
    print(json.dumps(compact,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
