"""Shared offline helpers; no network, model, account, or workspace dependency."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def setup(language='zh', serif=False):
    available = {font.name for font in font_manager.fontManager.ttflist}
    if serif:
        candidates = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
    elif language == 'en':
        candidates = ['Comic Sans MS', 'DejaVu Sans']
    else:
        candidates = ['Noto Sans CJK SC', 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    selected = next((name for name in candidates if name in available), None)
    if selected is None:
        raise RuntimeError('A CJK font is required. Install fonts-noto-cjk, then restart Python; see README.')
    plt.rcParams.update({'font.family': selected, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'figure.facecolor': 'white',
                         'axes.facecolor': 'white', 'font.size': 11})
    if serif:
        plt.rcParams['mathtext.fontset'] = 'stix'
    return selected


def save(fig, folder, stem='reproduced', tight=True, dpi=230):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    options = {'bbox_inches': 'tight', 'pad_inches': .18} if tight else {}
    for extension in ['png', 'svg']:
        fig.savefig(folder / (stem + '.' + extension), dpi=dpi, facecolor='white', **options)
    plt.close(fig)
