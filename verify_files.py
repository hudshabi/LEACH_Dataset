"""Verify this curated publication distribution; standard library only."""
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parent
if __name__=='__main__':
 count=0
 for line in (R/'SHA256SUMS.txt').read_text().splitlines():
  digest,name=line.split('  ',1);p=R/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit('Integrity failure: '+name)
  count+=1
 print('PASS:',count,'distribution files match SHA256 manifest')
