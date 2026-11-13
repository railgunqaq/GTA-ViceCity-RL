"""Check the selected Git index (or an unpacked source archive) before publishing."""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    '.zcode', '.codex', '.vscode', '.idea', '__pycache__', '.venv',
    'runs', 'shots', 'gamefiles', 'game-VC', 'build', 'bin', '.vs',
}
FORBIDDEN_SUFFIXES = {
    '.exe', '.dll', '.lib', '.pdb', '.obj', '.cso', '.pt', '.pth', '.npz',
    '.jsonl', '.out', '.log', '.zip', '.7z', '.rar', '.img', '.txd', '.dff',
    '.col', '.ipl', '.ide', '.gxt', '.scm', '.adf', '.raw', '.sdt', '.mp3',
    '.wav', '.pem', '.key', '.pfx', '.pyc',
}
PATTERNS = {
    'possible credential': re.compile(
        r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
        r'sk-[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16}|'
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'personal Windows path': re.compile(r'[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s"\']+'),
    'local provider configuration': re.compile(r'provider_config[.]json'),
}


def inventory():
    if (ROOT / '.git').exists():
        raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
        names = [part.decode('utf-8') for part in raw.split(b'\0') if part]
        if not names:
            raise RuntimeError('Git index is empty; select publication files with git add first.')
        modified = subprocess.check_output(['git', 'diff', '--name-only', '-z'], cwd=ROOT)
        if modified:
            raise RuntimeError('Tracked working files differ from the index; review and stage them first.')
        modes = subprocess.check_output(['git', 'ls-files', '--stage'], cwd=ROOT).decode('utf-8')
        if any(line.startswith(('160000 ', '120000 ')) for line in modes.splitlines()):
            raise RuntimeError('Unexpected submodule or symbolic link in publication index.')
        return names, 'git_index'
    names = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
             if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
    return sorted(names), 'archive_directory'


def main():
    try:
        names, scope = inventory()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({'passed': False, 'error': str(exc)}))
        return 1
    issues, total, largest = [], 0, 0
    for name in names:
        relative = Path(name)
        path = ROOT / relative
        reason = None
        if set(relative.parts) & FORBIDDEN_PARTS or relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            reason = 'excluded artifact type or directory'
        if relative.name.startswith('.env') and relative.name != '.env.example':
            reason = 'local environment file'
        if any('.pt.' in part or part.startswith('revc-ml-full-') for part in relative.parts):
            reason = 'model backup or bulk backup'
        if relative.name == 'building-collision-evidence.json':
            reason = 'full extracted game collision mesh'
        if reason:
            issues.append({'path': name, 'reason': reason})
        data = path.read_bytes()
        total += len(data)
        largest = max(largest, len(data))
        if len(data) > 10 * 1024 * 1024:
            issues.append({'path': name, 'reason': 'exceeds this source package 10 MiB/file budget'})
        if b'\0' in data:
            if relative.suffix.lower() not in {'.png', '.ico'}:
                issues.append({'path': name, 'reason': 'unexpected binary content'})
            continue
        try:
            content = data.decode('utf-8-sig')
        except UnicodeError:
            content = data.decode('latin-1')
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                issues.append({'path': name, 'reason': label})
    selected = set(names)
    required = ['README.md', 'NOTICE.md', 'ml/rl_sac.py', 'ml/ml_client.py',
                'ml/vc_task.py', 'reVC-src/src/ml/MlBridge.h',
                'reVC-src/vendor/librw/LICENSE', 'reVC-src/src/skel/win/gtavc.ico']
    required += [p.relative_to(ROOT).as_posix()
                 for p in (ROOT / 'reVC-src/src/extras/shaders/obj').glob('*.inc')]
    for name in required:
        if name not in selected:
            issues.append({'path': name, 'reason': 'required source missing from publication'})
    result = {'passed': not issues, 'scope': scope, 'files': len(names),
              'bytes': total, 'largest_file_bytes': largest, 'issues': issues}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(issues))


if __name__ == '__main__':
    sys.exit(main())
