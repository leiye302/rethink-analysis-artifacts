"""Data-equivalent centered bars; unknown actions remain in the remainder."""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting import rows, save, setup
ROOT = Path(__file__).resolve().parent
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced')
    args = parser.parse_args()
    setup()
    data = {row['mode']: row for row in rows(ROOT / 'data/counts.csv')}
    blue, purple = '#B3C8DE', '#E2D7EB'
    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    for y, mode in enumerate(['forbid', 'force', 'natural']):
        row = data[mode]
        percent = 100 * int(row['known_positive']) / int(row['n_generations'])
        ax.barh(y, percent, left=-percent, height=.39, color=blue, edgecolor='#7D9DBE', linewidth=.9)
        ax.barh(y, 100-percent, height=.39, color=purple, edgecolor='#B395CA', linewidth=.9, hatch='///')
        left_x = -percent/2 if percent > 14 else -percent-3
        ax.text(left_x, y, f'{percent:.2f}', va='center', ha='center' if percent > 14 else 'right', color='#36516C', weight='bold', fontsize=16)
        ax.text((100-percent)/2, y, f'{100-percent:.2f}', va='center', ha='center', color='#604C70', weight='bold', fontsize=16)
    ax.set_xlim(-100, 100)
    ax.set_ylim(2.6, -.6)
    ax.set_yticks([0, 1, 2], ['禁止触发', '强制触发', '自主触发'], fontsize=18)
    ax.tick_params(axis='y', length=0, pad=15)
    ax.set_xticks(list(range(-100, 101, 25)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{abs(value):g}'))
    ax.set_xlabel('占各模式全部生成的比例（%）；左右方向仅用于排版', fontsize=11)
    ax.axvline(0, color='#7D9DBE', lw=.8)
    ax.grid(axis='x', color='#E4E9F1', zorder=0)
    ax.set_axisbelow(True)
    for side in ['top', 'left', 'right']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#7D9DBE')
    fig.legend(handles=[Patch(facecolor=blue, edgecolor='#7D9DBE', label='已确认合理动作'),
                        Patch(facecolor=purple, edgecolor='#B395CA', hatch='///', label='其余动作（含未判定）')],
               ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.55, .87), fontsize=14)
    fig.suptitle('Bad reasoning 下的动作合理性', fontsize=24, weight='bold', y=.99)
    fig.subplots_adjust(left=.16, right=.97, bottom=.24, top=.70)
    fig.text(.16, .045, '13 个固定 Bad 状态 × 每状态 16 次采样 = 每模式 208 次生成；不是 208 个独立状态。\n其余动作含已确认不合理和未知，不能全部称为错误；不是完整 episode 成功率。', fontsize=9, color='#677383')
    save(fig, args.output_dir)


if __name__ == '__main__':
    main()
