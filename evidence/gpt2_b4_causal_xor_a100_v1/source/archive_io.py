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
