"""Match the user's compact blue/orange reference using all 2000 real points."""
import argparse
import sys
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from plotting import setup
PALETTE = ['#277DA8', '#D9772E']
WIDTH, HEIGHT = 8.0, 7.5
EPS = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reproduced')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = json.loads((ROOT/'raw/MEASUREMENTS_COMPLETE.json').read_text(encoding='utf-8'))
    path = ROOT/'raw/paired_results.jsonl'
    assert sha(path) == complete['results_sha256']
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == len({row['id'] for row in rows}) == 1000
    probabilities = [np.array([row['paired'][name]['target_probability'] for row in rows])
                     for name in ['without_marker', 'with_marker']]
    values = [np.log10(np.clip(q, EPS, 1-EPS)/(1-np.clip(q, EPS, 1-EPS))) for q in probabilities]
    previous = json.loads((ROOT/'data/visualization_audit.json').read_text(encoding='utf-8'))
    permutation = np.array(previous['display_permutation'])
    assert len(permutation) == 1000 and set(permutation) == set(range(1000))
    assert np.isclose(probabilities[0].mean(), previous['mean_Q_without'])
    assert np.isclose(probabilities[1].mean(), previous['mean_Q_with'])
    bandwidth = previous['density_bandwidth_in_z_space']
    setup()
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor='white')
    # Reference proportions: almost square cloud, a narrow density panel at right.
    ax = fig.add_axes([.111, .130, .620, .656])
    marginal = fig.add_axes([.7667, .130, .201, .656], sharey=ax)
    x = np.arange(1, 1001)
    xall = np.r_[x-.10, x+.10]
    yall = np.r_[values[0][permutation], values[1][permutation]]
    colors = np.array([PALETTE[0]]*1000+[PALETTE[1]]*1000)
    render_order = np.random.default_rng(2026).permutation(2000)
    points = ax.scatter(xall[render_order], yall[render_order], c=colors[render_order],
                        s=28, alpha=.80, edgecolors='white', linewidths=.30,
                        zorder=3, rasterized=True)
    assert len(points.get_offsets()) == 2000
    assert np.array_equal(np.asarray(points.get_offsets())[:, 1], yall[render_order])
    low, high = -6.25, 6.25
    assert np.all(yall >= low) and np.all(yall <= high)
    ax.set_yscale('function', functions=(np.arcsinh, np.sinh))
    ax.set_ylim(low, high)
    ax.set_yticks([-6, -4, -2, -1, 0, 1, 2, 4, 6])
    ax.set_xlim(0, 1001)
    ax.set_xticks([1, 200, 400, 600, 800, 1000])
    ax.set_xlabel('样本展示序号（固定随机顺序）', fontsize=10.0, labelpad=7)
    ax.set_ylabel(r'目标动作概率的对数几率 $z=\log_{10}[Q/(1-Q)]$', fontsize=10.7, labelpad=8)
    ax.axhline(0, color='#8993A3', lw=.9, ls=(0, (4, 4)), alpha=.6, zorder=1)
    ax.grid(axis='y', color='#E4E9F0', alpha=.8, linewidth=.65, zorder=0)
    for color, label in zip(PALETTE, ['未插入（1000 点）', '插入后（1000 点）']):
        ax.scatter([], [], color=color, s=34, alpha=.80, edgecolors='white', linewidths=.3, label=label)
    ax.legend(loc='lower left', bbox_to_anchor=(.015, 1.027), frameon=False,
              ncol=2, fontsize=9.7, columnspacing=1.6, handletextpad=.6)
    grid = np.linspace(low, high, 800)
    for z, color in zip(values, PALETTE):
        density = np.exp(-.5*((grid[:, None]-z[None, :])/bandwidth)**2).mean(axis=1)
        density /= bandwidth*np.sqrt(2*np.pi)
        marginal.fill_betweenx(grid, 0, density, color=color, alpha=.16)
        marginal.plot(density, grid, color=color, lw=2.1)
        marginal.scatter(np.full(len(z), -.007), z, marker='_', s=6,
                         linewidths=.4, color=color, alpha=.24, clip_on=False)
    marginal.set_xlim(left=0)
    marginal.set_xlabel('原始密度\n（z 空间）', fontsize=9.7, labelpad=7)
    marginal.set_title('原始分布', fontsize=12, pad=12, weight='bold')
    marginal.tick_params(axis='y', left=False, labelleft=False)
    for plot in [ax, marginal]:
        plot.tick_params(axis='both', length=3.5, pad=5, labelsize=9.5, colors='#657286')
        for side in ['top', 'right']:
            plot.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            plot.spines[side].set_color('#A9B3C1')
    marginal.spines['left'].set_visible(False)
    fig.text(.020, .978, '1000 条 SFT Bad：2000 个真实观测', fontsize=18.5,
             weight='bold', va='top', color='#242936')
    fig.text(.020, .902, '紧凑版 · 共用 asinh 纵轴 · 无重复扩增', fontsize=9.7,
             color='#657286', va='top')
    fig.text(.973, .902, f'均值 Q：{probabilities[0].mean():.3f} → {probabilities[1].mean():.3f}',
             fontsize=10.2, color='#657286', va='top', ha='right')
    fig.text(.020, .013,
             r'SFT 训练样本条件诊断，非任务成功率。纵轴刻度仍为 z；仅显示时将 Q 裁剪至 $[10^{-6},\,1-10^{-6}]$。',
             fontsize=7.2, color='#657286', va='bottom')
    fig.canvas.draw()
    size = [ax.get_position().width*WIDTH, ax.get_position().height*HEIGHT]
    assert np.isclose(size[0]/size[1], 1.008130081300813)
    stem = args.output_dir/'reproduced'
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        for suffix in ['.png', '.svg']:
            fig.savefig(stem.with_suffix(suffix), dpi=230, facecolor='white')
    missing = [str(w.message) for w in seen if 'Glyph' in str(w.message) and 'missing' in str(w.message)]
    assert not missing, missing
    plt.close(fig)
    assert sha(path) == complete['results_sha256']
    audit = {'states':1000, 'observations':2000, 'points_per_condition':1000,
        'repeated_points':0, 'data_changed_or_omitted':False,
        'source_results_sha256':sha(path), 'Q_without':float(probabilities[0].mean()),
        'Q_with':float(probabilities[1].mean()), 'figure_inches':[WIDTH,HEIGHT],
        'canvas_aspect_ratio':WIDTH/HEIGHT, 'main_axis_inches':size,
        'y_display_transform':'asinh(z), both groups identically; ticks label original z',
        'y_limits': [low,high], 'palette':dict(zip(['without_marker','with_marker'],PALETTE)),
        'same_density_bandwidth_as_previous':bandwidth, 'density_coordinate':'original log-odds z',
        'same_state_permutation_as_previous':True,'vertical_jitter':False,
        'marker_area':28,'marker_alpha':.80,'epsilon_for_display_only':EPS,
        'png_sha256':sha(stem.with_suffix('.png')),'svg_sha256':sha(stem.with_suffix('.svg')),
        'script_sha256':sha(Path(__file__))}
    (args.output_dir/'render_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
