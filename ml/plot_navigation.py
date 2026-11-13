"""Plot recorded navigation evidence without connecting to the game.

Usage: python ml/plot_navigation.py RUN_DIRECTORY [--output figure.png]
Requires matplotlib. --matplotlib-dir optionally adds an explicitly supplied
package directory, allowing plotting without changing the training environment.
The points are sampled collision contacts, not a reconstructed building mesh.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def point(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    try:
        value = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return value if all(math.isfinite(x) for x in value) else None


def json_lines(path):
    if not path.is_file():
        return
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A running producer can have a partial final line.
            if isinstance(row, dict):
                yield row


def read_evidence(directory):
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    routes = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(directory.glob('route-*.json'))]
    records = list(json_lines(directory / 'queries.jsonl'))
    if not records:
        records = [record for route in routes for record in route.get('checked_segments', [])]
    contacts, unavailable, frames = {}, {}, set()
    old_approach = None
    old_result = summary.get('old_direct_approach', {})
    for record in records:
        answer = record.get('result', record.get('record', record))
        if not isinstance(answer, dict):
            continue
        start, end = point(record.get('start', answer.get('start'))), point(record.get('end', answer.get('end')))
        if answer.get('frame') is not None:
            frames.add((answer.get('episode_id'), answer['frame']))
        if (old_result and answer.get('seq') == old_result.get('seq')
                and answer.get('frame') == old_result.get('frame') and start and end):
            old_approach = (start, end, answer)
        unknown = answer.get('status') == -3 or 'status=-3' in record.get('error', '')
        if unknown and start and end:
            midpoint = tuple((a + b) / 2 for a, b in zip(start, end))
            unavailable[tuple(round(v, 2) for v in midpoint)] = midpoint
        if answer.get('valid') is not True:
            continue
        for kind in ('line', 'sweep'):
            hit = answer.get(kind, {})
            pos = point(hit.get('point'))
            if hit.get('valid') is not True or hit.get('hit') is not True or pos is None:
                continue
            category = 'Building contacts' if hit.get('entity_type') == 1 else (
                'Object contacts' if hit.get('entity_type') == 4 else 'Other static contacts')
            key = tuple(round(value, 2) for value in pos) + (hit.get('entity_type'), hit.get('model_id'), hit.get('surface'))
            contacts[key] = dict(pos=pos, category=category, model_id=hit.get('model_id'), surface=hit.get('surface'))
    trace_states, motion = [], []
    for record in json_lines(directory / 'trace.jsonl'):
        state = record.get('after') or record.get('state')
        if isinstance(state, dict):
            if not trace_states:
                trace_states.append(record.get('before', state))
            pos = point((state.get('heli') or {}).get('pos'))
            if pos and (not motion or math.dist(pos, motion[-1]) > .01):
                motion.append(pos)
            if len(trace_states) == 1:
                trace_states.append(state)
            else:
                trace_states[-1] = state
    state = summary.get('survey_state') or summary.get('initial') or (trace_states[0] if trace_states else {})
    objectives = [item for item in state.get('objectives', [])
                  if item.get('valid') and item.get('kind') == 2 and point(item.get('pos'))]
    starts = [point(route.get('start')) for route in routes]
    starts = [pos for pos in starts if pos]
    if not starts:
        pos = point((state.get('heli') or {}).get('pos'))
        if pos:
            starts = [pos]
    unique_starts = list(dict.fromkeys(starts))
    return dict(summary=summary, routes=routes, records=records, contacts=list(contacts.values()),
                unavailable=list(unavailable.values()), frames=frames, old_approach=old_approach,
                objectives=objectives, starts=unique_starts, motion=motion)


def plot(directory, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    data = read_evidence(directory)
    if not (data['records'] or data['routes'] or data['motion']):
        raise ValueError('No query, route or trace evidence found in %s' % directory)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [1.8, 1.]})
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.24, hspace=.38)
    colors = {'Building contacts': '#56677c', 'Object contacts': '#d58638', 'Other static contacts': '#9573a4'}
    route_colors = ['#16804a', '#118ba5', '#9b6fc0', '#9b8700']
    old = data['old_approach']
    handles = {}
    all_points = [row['pos'] for row in data['contacts']] + data['starts'] + data['motion']
    all_points += [point(item['pos']) for item in data['objectives']]
    all_points += [pos for route in data['routes'] for value in route.get('path', []) if (pos := point(value))]
    if old:
        all_points.extend(old[:2])
    shared_x = None
    if all_points:
        lower, upper = min(p[0] for p in all_points), max(p[0] for p in all_points)
        pad = max(2., (upper - lower) * .04)
        shared_x = lower - pad, upper + pad
    for ax, vertical, name in zip(axes, (1, 2), ('Top view: XY', 'Height projection: XZ')):
        ax.set_facecolor('#fafbfc')
        ax.grid(color='#dce2e8', linewidth=.6, zorder=0)
        for category, color in colors.items():
            positions = [row['pos'] for row in data['contacts'] if row['category'] == category]
            if positions:
                ax.scatter([p[0] for p in positions], [p[vertical] for p in positions],
                           s=5, c=color, alpha=.45, edgecolors='none', label=category, zorder=2)
        if data['unavailable']:
            ax.scatter([p[0] for p in data['unavailable']], [p[vertical] for p in data['unavailable']],
                       s=23, marker='x', color='#a43fa1', linewidths=1.,
                       label='Unavailable query region (sample)', zorder=4)
        if old:
            start, end, answer = old
            ax.plot([start[0], end[0]], [start[vertical], end[vertical]], '--', color='#c73d3a',
                    linewidth=1.7, label='Old straight approach (blocked)', zorder=5)
            hit = answer.get('sweep', {})
            contact = point(hit.get('point'))
            if hit.get('hit') and contact:
                ax.scatter(contact[0], contact[vertical], s=75, marker='X', c='#c73d3a', zorder=8)
                if vertical == 1:
                    ax.annotate('Old approach contact\nmodel %s, surface %s' %
                                (hit.get('model_id'), hit.get('surface')),
                                (contact[0], contact[vertical]), xytext=(-12, 30),
                                textcoords='offset points', fontsize=9, ha='right',
                                bbox=dict(facecolor='white', edgecolor='#e0b2b2', alpha=.95, boxstyle='round,pad=.3'),
                                arrowprops=dict(arrowstyle='-', color='#c73d3a'))
        if data['motion']:
            ax.plot([p[0] for p in data['motion']], [p[vertical] for p in data['motion']],
                    color='#354ec3', linewidth=1.2, alpha=.8, label='Recorded physical motion', zorder=6)
        for route in data['routes']:
            index = int(route.get('objective_index', 0))
            path = [point(p) for p in route.get('path', [])]
            if route.get('status') == 'found' and len(path) >= 2 and all(path):
                ax.plot([p[0] for p in path], [p[vertical] for p in path], '-o',
                        color=route_colors[index % len(route_colors)], linewidth=2.1, markersize=3,
                        label='Validated route to D%d' % index, zorder=7)
            hover = point(route.get('goal'))
            if hover:
                ax.scatter(hover[0], hover[vertical], s=64, marker='o', facecolors='none',
                           edgecolors='#16804a', linewidths=1.2, label='Queried hover goal', zorder=7)
        for item in data['objectives']:
            pos, index = point(item['pos']), item['index']
            ax.scatter(pos[0], pos[vertical], s=65, marker='v', c='#b43c33',
                       edgecolors='white', linewidths=.6, label='Native drop destination', zorder=9)
            ax.annotate('D%d' % index, (pos[0], pos[vertical]), xytext=(7, -3),
                        textcoords='offset points', color='#873027', weight='bold', fontsize=10)
        for pos in data['starts']:
            ax.scatter(pos[0], pos[vertical], s=125, marker='*', c='#174c92',
                       edgecolors='white', linewidths=.6, label='Planning start', zorder=10)
        ax.set_title(name, loc='left', fontsize=12, weight='bold')
        ax.set_xlabel('World X (m)')
        ax.set_ylabel('World %s (m)' % ('Y' if vertical == 1 else 'Z'))
        ax.margins(x=.04, y=.12)
        if shared_x:
            ax.set_xlim(shared_x)
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = handle
    status_parts = ['D%s: %s' % (r.get('objective_index', '?'), r.get('status', 'unknown')) for r in data['routes']]
    state_label = ' | '.join(status_parts) if status_parts else 'Routes not yet written'
    fig.suptitle('Navigation evidence — %s' % directory.name, fontsize=16, weight='bold', x=.075, ha='left', y=.97)
    fig.text(.075, .925, 'Sampled collision contacts only — not a complete building map', fontsize=12, color='#5c6570')
    fig.text(.075, .894, state_label, fontsize=10, color='#394657')
    fig.legend(list(handles.values()), list(handles.keys()), loc='lower center',
               bbox_to_anchor=(.52, .065), ncol=3, frameon=False, fontsize=9, markerscale=1.4)
    fig.text(.075, .032, '%s native queries; %s distinct sampled contacts; %s frozen snapshot(s). '
             'Projection overlays different heights/rooms.' %
             (len(data['records']), len(data['contacts']), len(data['frames'])), fontsize=9, color='#555e6a')
    fig.text(.075, .013, 'Routes, if shown, were checked against then-loaded static geometry. '
             'Dynamic obstacles and future flight tracking are not certified.', fontsize=9, color='#555e6a')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white', metadata={'Title': 'Recorded navigation evidence',
                'Description': 'Sampled native collision contacts, not a building mesh or certified flight path.'})
    plt.close(fig)
    return dict(output=str(output.resolve()), query_records=len(data['records']),
                contacts=len(data['contacts']), unavailable_samples=len(data['unavailable']),
                native_destinations=len(data['objectives']),
                validated_routes=sum(r.get('status') == 'found' for r in data['routes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--matplotlib-dir', type=Path)
    args = parser.parse_args()
    if args.matplotlib_dir:
        sys.path.insert(0, str(args.matplotlib_dir.resolve()))
    output = args.output or args.directory / 'navigation-evidence.png'
    print(json.dumps(plot(args.directory.resolve(), output.resolve()), indent=2))
