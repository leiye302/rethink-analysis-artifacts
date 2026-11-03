"""Offline checks of file hashes, per-token means, paired probabilities and counts.

Standard library only. Checks raise exceptions (not disabled by python -O).
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def require(condition, description):
    if not condition:
        raise ValueError(description)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    hashes = (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    for line in hashes:
        digest, name = line.split('  ', 1)
        require(sha(ROOT/name) == digest, 'File hash mismatch: '+name)
    original = load(ROOT/'artifact_manifest.json')['files']
    for item in original:
        require(sha(ROOT/item['path']) == item['sha256'], 'Source-copy hash mismatch: '+item['path'])
    # 01: region means from each measured token, not pre-aggregated-only data.
    folder = ROOT/'01_baseline_signal_distribution'
    values = rows(folder/'data/decision_values.csv')
    require(len(values) == len({v['id'] for v in values}) == 200, '01 paired-state count')
    for row in values:
        score = load(folder/'raw/scores'/f"{row['id']}.json")
        tokens = score['conditions']['native']['tokens']
        for region in ['reasoning', 'action']:
            selected = [t for t in tokens if t['region'] == region]
            require(len(selected) == int(row[region+'_tokens']), '01 region token count')
            require(all(abs(abs(t['teacher_logp']-t['student_logp'])-t['abs_gap']) < 1e-6 for t in selected),
                    '01 float32 subtraction parity')
            require(abs(statistics.mean(t['abs_gap'] for t in selected)-float(row[region])) < 1e-12,
                    '01 per-decision mean')
    means1 = {r: statistics.mean(float(v[r]) for v in values) for r in ['action', 'reasoning']}
    require(abs(means1['action']-.034637060828157686) < 1e-12, '01 action mean')
    require(abs(means1['reasoning']-.2340344554144021) < 1e-12, '01 reasoning mean')
    require(sum(0 <= float(v['action']) < .02 for v in values) == 149, '01 first bin')
    # 02: same reviewed IDs across all six cells, actual generated reference hits.
    folder = ROOT/'02_reasoning_intervention'
    scores = lines(folder/'raw/score.jsonl')
    cases = {c['id']: c for c in lines(folder/'raw/paired_reviewed_cases.jsonl')}
    require(len(scores) == len(cases) == 22 and {s['id'] for s in scores} == set(cases), '02 state identities')
    expected = {'teacher_correct': {'teacher': 19, 'student': 17},
                'deleted': {'teacher': 13, 'student': 13}, 'student_wrong': {'teacher': 10, 'student': 9}}
    cells = rows(folder/'data/decision_cells.csv')
    require(len(cells) == 132, '02 132 cells')
    cell_map = {(c['id'], c['condition'], c['branch']): c for c in cells}
    require(len(cell_map) == 132, '02 unique cells')
    for count in rows(folder/'data/counts.csv'):
        condition, branch = count['condition'], count['branch']
        hits = 0
        for score in scores:
            case = cases[score['id']]
            cell = score['cells'][condition][branch]
            action = cell['free_action']
            hit = action['action'] in case['correct_actions']
            require(hit == action['correct'], '02 reference-set hit parity')
            require(cell_map[(score['id'], condition, branch)]['generated_action'] == (action['action'] or ''), '02 CSV action parity')
            mass = sum(cell['actions'][a]['candidate_probability'] for a in case['correct_actions'])
            require(abs(mass-cell['correct_probability']) < 1e-10, '02 candidate Q mass')
            hits += hit
        require(hits == int(count['hits']) == expected[condition][branch], '02 six-cell counts')
    # 03: all 1000 true pairs, all candidate softmax values, original single target.
    folder = ROOT/'03_rethink_insertion'
    raw = lines(folder/'raw/paired_results.jsonl')
    samples = {c['sample_id']: c for c in lines(folder/'raw/sampled_sft_bad1000.jsonl')}
    require(len(raw) == len({s['id'] for s in raw}) == len(samples) == 1000, '03 sample count')
    require({s['id'] for s in raw} == set(samples), '03 sample identities')
    csvpairs = {r['sample_id']: r for r in rows(folder/'data/paired_Q1000.csv')}
    require(len(csvpairs) == 1000, '03 CSV pair count')
    observations = rows(folder/'data/observations_2000.csv')
    require(len(observations) == 2000, '03 observation count')
    masses = {'without_marker': [], 'with_marker': []}
    for record in raw:
        sample = samples[record['id']]
        require(record['reference_action'] == sample['oracle_action'], '03 stored target unchanged')
        arms = record['paired']
        without, with_marker = [arms[k]['prefilled_suffix_ids'] for k in masses]
        require(with_marker == [151665]+without, '03 exactly one inserted token')
        require(arms['with_marker']['scoring_boundary'] == arms['without_marker']['scoring_boundary']+1, '03 boundary')
        for arm in masses:
            cell = arms[arm]
            actions = cell['action_scores']
            require(set(actions) == set(sample['admissible_actions_before']), '03 same admissible set')
            peak = max(a['log_probability'] for a in actions.values())
            denominator = sum(math.exp(a['log_probability']-peak) for a in actions.values())
            for action in actions.values():
                probability = math.exp(action['log_probability']-peak)/denominator
                require(abs(probability-action['candidate_probability']) < 1e-10, '03 candidate softmax')
            target = actions[sample['oracle_action']]['candidate_probability']
            require(abs(target-cell['target_probability']) < 1e-12, '03 single target Q')
            field = 'Q_without' if arm == 'without_marker' else 'Q_with'
            require(abs(float(csvpairs[record['id']][field])-target) < 1e-14, '03 CSV Q parity')
            masses[arm].append(target)
    before, after = masses.values()
    means3 = [statistics.mean(v) for v in [before, after]]
    require(abs(means3[0]-.12082768806300861) < 1e-12, '03 before mean')
    require(abs(means3[1]-.6118690232933994) < 1e-12, '03 after mean')
    increase = sum(b > a for a, b in zip(before, after))
    require(increase == 918, '03 increase count')
    display = load(folder/'data/visualization_audit.json')
    require(sorted(display['display_permutation']) == list(range(1000)), '03 display permutation')
    # 04: retain both original labels and existing final-reviewed labels.
    folder = ROOT/'04_bad_reasoning_action_quality'
    panel = {r['decision_id']: r for r in lines(folder/'raw/panel_final_adjudication.jsonl')}
    require(len(panel) == 13, '04 states')
    counts = collections.defaultdict(collections.Counter)
    repeats = collections.defaultdict(set)
    jobs = list((folder/'raw/jobs').glob('*.json'))
    require(len(jobs) == 624, '04 raw generation count')
    for path in jobs:
        r = load(path)['result']
        state = panel[r['decision_id']]
        require(state['evaluation']['reasoning_label'] == 'bad', '04 Bad subset')
        action = r.get('action')
        if r.get('failure'):
            value = None
        elif not action or action not in state['evaluation']['admissible_actions']:
            value = 0
        else:
            value = {a['text']: a['value'] for a in state['evaluation']['actions']}.get(action)
        label = 'positive' if value == 1 else 'unknown' if value is None else 'zero'
        counts[r['mode']][label] += 1
        repeats[(r['decision_id'], r['mode'])].add(r['repeat'])
    require(len(repeats) == 39 and all(len(v) == 16 for v in repeats.values()), '04 16 distinct repeats')
    for row in rows(folder/'data/counts.csv'):
        count = counts[row['mode']]
        for key, field in [('positive', 'known_positive'), ('zero', 'known_zero'), ('unknown', 'unknown')]:
            require(count[key] == int(row[field]), '04 final adjudication count')
        require(sum(count.values()) == int(row['n_generations']) == 208, '04 denominator')
    require([counts[m]['positive'] for m in ['forbid', 'force', 'natural']] == [17, 164, 64], '04 positives')
    report = {'passed': True, 'sha256_files_checked': len(hashes),
              '01': {'paired_decisions': 200, 'means': means1}, '02': {'states': 22, 'cells': 132, 'hits': expected},
              '03': {'paired_samples': 1000, 'real_observations': 2000, 'mean_Q': means3, 'increased': increase, 'decreased': 1000-increase},
              '04': {'states': 13, 'generations_per_mode': 208, 'total': 624, 'counts': dict(counts)},
              'network_calls': 0, 'model_inference_calls': 0}
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
