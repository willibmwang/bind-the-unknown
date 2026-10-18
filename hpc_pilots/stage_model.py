"""Stage a pinned, ungated instruction-model snapshot without executing model code."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .common import ROOT, digest, write_json


def main():
    repo = 'Qwen/Qwen3-4B-Instruct-2507'
    manifest_path = ROOT / 'models/manifest.json'
    revision = (json.loads(manifest_path.read_text())['revision'] if manifest_path.exists()
                else HfApi().model_info(repo).sha)
    target = ROOT / 'models/qwen3-4b'
    write_json(manifest_path, {'repo': repo, 'revision': revision, 'status': 'downloading',
                              'license': 'Apache-2.0'})
    snapshot_download(repo_id=repo, revision=revision, local_dir=target,
                      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
    files = {str(p.relative_to(target)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(target.iterdir()) if p.is_file()}
    write_json(manifest_path, {'repo': repo, 'revision': revision, 'status': 'staged',
                              'license': 'Apache-2.0', 'files': files,
                              'remote_code_execution': False})
    print(json.dumps({'model': repo, 'revision': revision, 'files': len(files), 'status': 'staged'}))


if __name__ == '__main__':
    main()
