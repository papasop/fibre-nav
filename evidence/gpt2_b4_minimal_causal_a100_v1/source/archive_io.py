import hashlib,json,zipfile
from pathlib import Path
def restore_archive(path,destination):
    destination=Path(destination)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
        for n in names:
            p=Path(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n:raise ValueError('Unsafe archive path')
        if z.testzip():raise ValueError('ZIP CRC failure')
        manifest=json.loads(z.read('MANIFEST.json'))
        if set(names)-set(manifest)-{'MANIFEST.json'}:raise ValueError('Unhashed archive member')
        for n,h in manifest.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Archive digest mismatch: '+n)
        identity=json.loads(z.read('identity.json'))
        from protocol import PROTOCOL
        if identity['protocol']!=PROTOCOL:raise ValueError('Protocol mismatch')
        for n in names:
            p=destination/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
    return identity

def archive_progress(root):
    root=Path(root)
    complete=sum(1 for p in (root/'seeds').glob('*/SUMMARY.json'))
    phases=sum(len(json.loads(p.read_text())) for p in (root/'seeds').glob('*/records.json'))
    progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
    records=root/'seeds'/str(progress.get('seed',''))/'records.json'
    completed=json.loads(records.read_text()) if records.exists() else {}
    steps=0 if progress.get('phase') in completed else progress.get('steps',0)
    return complete,phases,steps
