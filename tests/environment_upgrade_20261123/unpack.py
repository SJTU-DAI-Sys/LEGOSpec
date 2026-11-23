from pathlib import Path
import urllib.request,json,zipfile
work=Path('tests/environment_upgrade_20261123');j=json.load(urllib.request.urlopen('https://pypi.org/pypi/sglang/0.5.21/json',timeout=30))
url=next(f['url'] for f in j['urls'] if f['filename'].endswith('.whl') and ('none-any' in f['filename'] or ('cp311' in f['filename'] and 'x86_64' in f['filename'])))
p=work/'sglang-0.5.21.whl';urllib.request.urlretrieve(url,p)
with zipfile.ZipFile(p) as z:z.extractall(work/'upstream0521')
print('extracted',p.stat().st_size)
