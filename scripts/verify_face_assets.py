#!/usr/bin/env python3
"""Verify the separately distributed face recordings, statistics and weights."""

import argparse
import hashlib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root', type=Path, default=PROJECT_DIR,
        help='directory containing the extracted datasets/ and ckpts/ folders',
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = PROJECT_DIR / 'configs' / 'face_assets.sha256'
    checked = 0
    failures = []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            failures.append(f'Missing file or unexpected symlink: {relative}')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            failures.append(f'Checksum mismatch: {relative}')
        checked += 1
    if failures:
        failures.append(
            'Extract echoavatar_face_training_assets.tar.gz at the repository root. '
            'See readme.md#3-train-the-face-motion-generator for setup instructions. '
            'This check applies to the supplied assets, not a replacement dataset.'
        )
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {checked} face asset files match their SHA256 checksums.')


if __name__ == '__main__':
    main()
