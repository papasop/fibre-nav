"""Read-only CPU compact export. Paste into Colab; no model, Torch or archived code execution."""
from pathlib import Path, PurePosixPath
import hashlib, json, tempfile, time, zipfile, shutil

PINS = {
 'c154': ('GPT2_LC2_C154_FIXED_DIRECTION_CPU_V1_RECOVERY/c154_fixed_direction_1792741096713648254_results.zip', '25d79b8ef8e020fa91fa13fad4b220c2f6e7d471f411254db390f855a392d11e'),
 'c155': ('GPT2_LC2_C155_PAIRED_FORWARD_PRECISION_CPU_V1_RECOVERY/c155_paired_forward_precision_1792746742162409955_results.zip', '5e52577034889cc7b5d0e0d77382d52a8dc87e0e608408f72de040ef5f14eb8b')}
PREFIX = 'payload/evidence/gpt2_lc2_c141_c155/supplement/'

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def safe(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:
        raise ValueError('Unsafe ZIP member: '+name)

def export_one(path, expected, out, stage):
    if sha_file(path)!=expected:raise ValueError('Original ZIP SHA256 mismatch: '+stage)
    with zipfile.ZipFile(path) as z:
        names=[i.filename for i in z.infolist() if not i.is_dir()]
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member')
        for n in names:safe(n)
        if 'RESULT_MANIFEST.json' not in names:raise ValueError('Missing root RESULT_MANIFEST.json')
        manifest=json.loads(z.read('RESULT_MANIFEST.json'))
        if set(manifest)!=set(names)-{'RESULT_MANIFEST.json'}:raise ValueError('Original manifest inventory mismatch')
        exported={};omitted={};counts={}
        for n in names:
            h=hashlib.sha256()
            with z.open(n) as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
            digest=h.hexdigest()
            if n!='RESULT_MANIFEST.json' and digest!=manifest[n]:raise ValueError('Original member hash mismatch: '+n)
            # Keep ALL text records, all measurements, source and small frozen tensors.
            # Omit only large NPZ arrays, never individual failed measurement rows.
            info=z.getinfo(n)
            if n.endswith('.npz') and info.file_size>20*1024*1024:
                omitted[n]={'sha256':digest,'size_bytes':info.file_size,'reason':'large_tensor_not_in_compact_supplement'}
                continue
            raw=z.read(n);out.writestr(PREFIX+stage+'/'+n,raw)
            exported[n]={'sha256':digest,'size_bytes':len(raw)}
            if n.endswith('.jsonl'):counts[n]=len(raw.splitlines())
        summary=json.loads(z.read('SUMMARY.json'))
        if summary.get('complete') is not True or summary.get('LC2_closed') is not False:
            raise ValueError('Unexpected original completion/LC2 status')
    if sha_file(path)!=expected:raise ValueError('Original changed during export')
    return dict(original_filename=Path(path).name,original_sha256=expected,
        original_bytes_rehashed=True,all_original_manifest_members_hashed=True,
        exported_members=exported,omitted_members=omitted,jsonl_record_counts=counts,
        original_decision=summary.get('decision'),new_model_forwards=0,
        archived_source_executed=False,model_measurements_independently_recomputed=False,
        gradients_independently_verified=False,LC2_closed=False)

def main():
    drive=Path('/content/drive/MyDrive')
    if not drive.is_dir():
        from google.colab import drive as colab_drive
        colab_drive.mount('/content/drive')
    target=Path('/content')/('LC2_C154_C155_COMPACT_SUPPLEMENT_'+str(time.time_ns())+'.zip')
    reports={}
    try:
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
            for stage,(rel,pin) in PINS.items():
                print('READ_ONLY_EXPORT',stage,flush=True)
                reports[stage]=export_one(drive/rel,pin,out,stage)
            out.writestr(PREFIX+'SUPPLEMENT_PROVENANCE.json',json.dumps(reports,ensure_ascii=False,indent=2))
        dest=drive/'GPT2_LC2_C154_C155_GITHUB_SUPPLEMENT_RECOVERY'/target.name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dest)
        if sha_file(dest)!=sha_file(target):raise ValueError('Backup differs')
        print('SUPPLEMENT_COMPLETE',target,'SHA256',sha_file(target),flush=True)
        print('Original chronology, measurement accuracy and non-access are not independently proved; LC2 remains open.')
        from google.colab import files
        files.download(str(target))
    except BaseException:
        # A partially written supplement must never look like a successful export.
        if target.exists():target.rename(target.with_suffix('.incomplete.zip'))
        raise

if __name__=='__main__':main()
