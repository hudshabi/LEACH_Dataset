"""Restore archived original scored outputs; standard library only."""
from pathlib import Path
from zipfile import ZipFile
import json,hashlib
R=Path(__file__).resolve().parent
if __name__=='__main__':
 count=0
 for entry in json.loads((R/'raw_round_logs/index.json').read_text()):
  p=R/entry['archive']
  if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:raise SystemExit('Archive hash mismatch: '+str(p))
  with ZipFile(p) as z:
   for info in z.infolist():
    target=(R/info.filename).resolve()
    if not target.is_relative_to((R/'extension').resolve()):raise SystemExit('Unsafe archive path')
   z.extractall(R);count+=len(z.infolist())
 print('Restored',count,'original scored-output files under extension/')
