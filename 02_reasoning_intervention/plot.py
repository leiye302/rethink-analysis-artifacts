"""Data-equivalent dual rings; supplied screenshot's exact styling source unknown."""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting import rows, save, setup
ROOT = Path(__file__).resolve().parent
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced')
    args = parser.parse_args()
    setup(language='en')
    data = rows(ROOT / 'data/counts.csv')
    blue, purple = '#ABC4DE', '#DDD0E7'
    ink = '#2F455E'
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    names = {'teacher_correct': 'Correct\nReasoning', 'deleted': 'No\nReasoning', 'student_wrong': 'Wrong\nReasoning'}
    for ax, condition in zip(axes, names):
        branches = {row['branch']: row for row in data if row['condition'] == condition}
        for branch, radius, width in [('teacher', 1.03, .22), ('student', .72, .39)]:
            row = branches[branch]
            hit, n = int(row['hits']), int(row['n'])
            wedges, *_ = ax.pie([hit, n - hit], radius=radius, startangle=90, counterclock=False,
                               colors=[blue, purple], wedgeprops={'width': width, 'edgecolor': 'white', 'linewidth': 1})
            if branch == 'teacher':
                for wedge in wedges:
                    wedge.set_hatch('///')
                    wedge.set_edgecolor('#8198B3' if wedge is wedges[0] else '#B092C7')
            if branch == 'student':
                import math
                angle = math.radians(90 - 180 * hit / n)
                x, y = .56 * math.cos(angle), .56 * math.sin(angle)
                ax.text(x, y, f'{100 * hit/n:.1f}%', ha='center', va='center', fontsize=12.5, color=ink, weight='bold')
                ax.text(-x, -y, f'{100 * (n-hit)/n:.1f}%', ha='center', va='center', fontsize=12.5, color='#71518B')
        t = branches['teacher']
        s = branches['student']
        ax.text(0, 0, names[condition], ha='center', va='center', fontsize=11.5, weight='bold', color='#283746')
        ax.text(-.94, 1.04, f"{100*(1-int(t['hits'])/int(t['n'])):.1f}%", color='#71518B', fontsize=14, ha='left', weight='bold')
        ax.text(.94, 1.04, f"{100*int(t['hits'])/int(t['n']):.1f}%", color=ink, fontsize=14, ha='right', weight='bold')
        ax.text(0, -1.22, f"Teacher: {t['hits']}/{t['n']}   Student: {s['hits']}/{s['n']}", ha='center', fontsize=10, color='#5C6671')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.35, 1.2)
    fig.legend(handles=[Patch(facecolor=blue, label='Reference hit'), Patch(facecolor=purple, label='Not a reference hit'),
                        Patch(facecolor='white', edgecolor='#8198B3', hatch='///', label='Teacher: outer'),
                        Patch(facecolor='white', edgecolor='#8198B3', label='Student: inner')],
               loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1), fontsize=11)
    fig.subplots_adjust(top=.83, bottom=.14, left=.02, right=.98, wspace=.04)
    fig.text(.5, .025, '22 paired states; next-action reference-set hit, NOT episode success.', ha='center', fontsize=10, color='#5C6671')
    save(fig, args.output_dir)


if __name__ == '__main__':
    main()
