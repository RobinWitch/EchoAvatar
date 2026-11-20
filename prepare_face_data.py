#!/usr/bin/env python3
"""Inspect paired LiveLinkFace recordings and run the training preprocessing path."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import wave

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'datasets' / 'face_livelinkface'


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def inspect_recordings(data_dir):
    csv_files = sorted((data_dir / 'arkit').glob('*.csv'))
    wav_files = sorted((data_dir / 'audio').glob('*.wav'))
    csv_names = {p.stem for p in csv_files}
    wav_names = {p.stem for p in wav_files}
    if not csv_names or csv_names != wav_names:
        raise ValueError(
            f'Expected nonempty, one-to-one CSV/WAV pairs. '
            f'Missing WAV: {sorted(csv_names - wav_names)}; '
            f'missing CSV: {sorted(wav_names - csv_names)}'
        )
    pairs = []
    reference_header = None
    sample_rates = Counter()
    for csv_path in csv_files:
        with csv_path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if reference_header is None:
                reference_header = header
            if header != reference_header or len(header[2:-9]) != 52:
                raise ValueError(f'Unexpected CSV header: {csv_path}')
            frames = 0
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f'Invalid CSV row {frames + 2}: {csv_path}')
                frames += 1
        wav_path = data_dir / 'audio' / f'{csv_path.stem}.wav'
        with wave.open(str(wav_path), 'rb') as audio:
            rate = audio.getframerate()
            seconds = audio.getnframes() / rate
            channels = audio.getnchannels()
        if frames == 0 or seconds <= 0:
            raise ValueError(f'Empty recording: {csv_path.stem}')
        sample_rates[rate] += 1
        pairs.append({
            'name': csv_path.stem,
            'audio': wav_path.relative_to(data_dir).as_posix(),
            'arkit': csv_path.relative_to(data_dir).as_posix(),
            'audio_sample_rate': rate,
            'audio_channels': channels,
            'audio_seconds': seconds,
            'raw_arkit_frames': frames,
            'duration_difference_seconds': seconds - frames / 60.0,
        })
    return {
        'dataset_name': data_dir.name,
        'recording_pairs': len(pairs),
        'total_audio_seconds': sum(p['audio_seconds'] for p in pairs),
        'total_raw_arkit_frames': sum(p['raw_arkit_frames'] for p in pairs),
        'raw_csv_fps_assumed_by_original_loader': 60,
        'training_fps': 30,
        'training_audio_sample_rate': 16000,
        'audio_sample_rate_counts': dict(sample_rates),
        'blendshape_names_in_training_order': reference_header[2:-9],
        'augmentation_speed_factors': [0.9, 1.0, 1.1],
        'augmented_sequences': len(pairs) * 3,
        'dataset_length_after_repeat': len(pairs) * 3 * 10,
        'window_size_frames': 64,
        'pairs': pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--metadata-only', action='store_true',
                        help='check raw CSV/WAV metadata using only the Python standard library')
    parser.add_argument('--recompute-stats', action='store_true',
                        help='replace arkit_mean_std.npy with statistics computed from this dataset')
    parser.add_argument('--report', help='optional JSON report path, relative to the repository')
    args = parser.parse_args()
    if args.metadata_only and args.recompute_stats:
        parser.error('--recompute-stats cannot be combined with --metadata-only')
    data_dir = resolve_path(args.data_dir)
    report = inspect_recordings(data_dir)
    print(f"Paired recordings: {report['recording_pairs']}")
    print(f"Audio duration: {report['total_audio_seconds'] / 60:.2f} minutes")
    print('Processing: 16 kHz mono audio; 30 fps, 52 blendshape channels; speed factors 0.9/1.0/1.1')

    if not args.metadata_only:
        import numpy as np
        import torch
        from utils.face_dataset import ARKitAudioDataset

        dataset = ARKitAudioDataset(data_dir=str(data_dir),
                                   make_new_mean_std=args.recompute_stats)
        np.random.seed(123)
        for idx in range(dataset.origin_len):
            sample = dataset[idx]
            if sample['arkit'].shape != (64, 52) or sample['audio'].shape != (34133,):
                raise ValueError(f'Unexpected sample shape at index {idx}')
            if not torch.isfinite(sample['audio']).all() or not torch.isfinite(sample['arkit']).all():
                raise ValueError(f'Non-finite values at sample {idx}')
        print(f'Processed sequences: {dataset.origin_len}; dataset length: {len(dataset)}')
        print('Every augmented sequence passed an aligned crop and finite-value check.')
        report['preprocessing_checked'] = True
        report['sample_audio_shape'] = [34133]
        report['sample_arkit_shape'] = [64, 52]

    if args.report:
        report_path = resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Report: {report_path}')


if __name__ == '__main__':
    main()
