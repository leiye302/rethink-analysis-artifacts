"""Original histogram design and adaptive reflected-KDE smoothing, offline."""
import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting import rows, save, setup
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent


def adaptive_bandwidth(values):
    q25, q75 = np.quantile(values, [.25, .75])
    scale = min(float(np.std(values, ddof=1)), float((q75-q25)/1.349))
    pilot = 2.5*.9*scale*len(values)**(-.2)
    z = (values[:, None]-values[None, :])/pilot
    zr = (values[:, None]+values[None, :])/pilot
    density = (np.exp(-.5*z*z)+np.exp(-.5*zr*zr)).mean(axis=1)/(pilot*math.sqrt(2*math.pi))
    return pilot*np.sqrt(np.exp(np.log(density).mean())/density)


def cdf(x):
    a = np.asarray(x, dtype=float)
    values = np.fromiter((math.erf(v/math.sqrt(2)) for v in a.ravel()), dtype=float, count=a.size)
    return .5*(1+values.reshape(a.shape))


def bin_percent(x, values, bandwidths, width):
    left = np.maximum(x[:, None]-width/2, 0)
    right = x[:, None]+width/2
    positive = cdf((right-values)/bandwidths)-cdf((left-values)/bandwidths)
    reflected = cdf((right+values)/bandwidths)-cdf((left+values)/bandwidths)
    return np.maximum(100*(positive+reflected).mean(axis=1), 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE/'reproduced')
    args = parser.parse_args()
    setup(language='en', serif=True)
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 18,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 16,
                         'text.color': '#202531', 'axes.labelcolor': '#202531'})
    records = rows(HERE/'data/decision_values.csv')
    r, a = [np.array([float(row[k]) for row in records]) for k in ['reasoning', 'action']]
    assert len(r) == len(a) == 200
    width = .02
    upper = np.ceil(max(r.max(), a.max())/width)*width
    edges = np.linspace(0, upper, int(round(upper/width))+1)
    pr, pa = [np.histogram(v, edges)[0]/len(v)*100 for v in [r, a]]
    assert np.isclose(pr.sum(), 100) and np.isclose(pa.sum(), 100)
    x = np.linspace(width/2, upper, 2400)
    yr, ya = [bin_percent(x, v, adaptive_bandwidth(v), width) for v in [r, a]]
    fig, (high, low) = plt.subplots(2, 1, sharex=True, figsize=(8.8, 5.9),
                                  gridspec_kw={'height_ratios': [.75, 3.6], 'hspace': .09})
    fig.subplots_adjust(left=.125, right=.975, bottom=.185, top=.91)
    for ax in [high, low]:
        centers = (edges[:-1]+edges[1:])/2
        widths = np.diff(edges)*.94
        ax.bar(centers, pr, width=widths, color='#B5BEC8', edgecolor='white', linewidth=.45, alpha=.87, zorder=2)
        ax.bar(centers, pa, width=widths, color='#8866AA', edgecolor='white', linewidth=.45, alpha=.83, zorder=3)
        ax.axvline(a.mean(), color='#634485', lw=1.65, ls=(0, (4, 3)), zorder=4)
        ax.axvline(r.mean(), color='#737F8E', lw=1.65, ls=(0, (4, 3)), zorder=4)
        ax.plot(x, yr, color='#566271', lw=2.35, zorder=8, solid_capstyle='round')
        ax.plot(x, ya, color='#5E348A', lw=2.35, zorder=9, solid_capstyle='round')
        ax.set_xlim(edges[0], edges[-1])
        ax.xaxis.set_major_locator(MultipleLocator(.1))
        ax.tick_params(direction='out', length=5, width=.9, pad=5)
        for spine in ax.spines.values():
            spine.set_linewidth(.9)
            spine.set_color('#414650')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E9EDF2', lw=.65)
    high.set_ylim(71, 78.3)
    high.set_yticks([72, 76])
    low.set_ylim(0, 17.6)
    low.set_yticks([0, 5, 10, 15])
    high.spines['bottom'].set_visible(False)
    low.spines['top'].set_visible(False)
    high.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for ax, y in [(high, 0), (low, 1)]:
        ax.plot([0, 1], [y, y], transform=ax.transAxes, marker=[(-1, -.45), (1, .45)],
                markersize=9, linestyle='none', color='#414650', mew=.95, clip_on=False, zorder=10)
    high.annotate(f'{pa.max():.1f}%', xy=(width/2, pa.max()), xytext=(.063, 75.8),
                  color='#634485', fontsize=14, fontweight='bold', ha='left', va='center',
                  arrowprops={'arrowstyle': '-', 'color': '#634485', 'lw': 1}, zorder=10)
    low.legend(handles=[Patch(facecolor='#8866AA', alpha=.83), Patch(facecolor='#B5BEC8', alpha=.87)],
               labels=[r'Action  ($\mu$='+f'{a.mean():.4f})', r'Reasoning  ($\mu$='+f'{r.mean():.4f})'],
               loc='upper right', frameon=False, handlelength=1.4, handletextpad=.6, labelspacing=.55)
    fig.supylabel('Percentage (%)', x=.032, y=.55, fontsize=19)
    low.set_xlabel(r'Per-decision mean $|\log p_T-\log p_S|$ (nat)', labelpad=10)
    fig.text(.125, .955, 'Baseline signal distributions', fontsize=20, fontweight='bold')
    fig.text(.975, .956, '200 decisions per region', fontsize=12, ha='right', color='#5E6875')
    fig.text(.125, .035, 'Bins: 0.02 nat. Solid curves: smoothed bin probabilities. Dashed lines: means. Y-axis break marked.',
             fontsize=10.5, color='#5E6875')
    save(fig, args.output_dir, tight=False, dpi=220)
    print('01: 200 paired decisions; all bars and means retained.')


if __name__ == '__main__':
    main()
