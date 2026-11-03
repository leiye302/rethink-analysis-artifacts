"""Regenerate all four figures into a separate output directory."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ['01_baseline_signal_distribution', '02_reasoning_intervention',
               '03_rethink_insertion', '04_bad_reasoning_action_quality']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for name in EXPERIMENTS:
        subprocess.run([sys.executable, str(ROOT / name / 'plot.py'),
                        '--output-dir', str(output / name)], check=True)
    print('All four figures regenerated; original published figures were not overwritten.')


if __name__ == '__main__':
    main()
