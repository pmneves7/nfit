"""Isolated SEQUOIA detector-normalization lifecycle benchmark."""
from __future__ import annotations

import argparse
import json
import resource
import socket
import sys
import time
import zipfile
from types import SimpleNamespace

import numpy as np


def prepare(project, destination):
    from nfit.mdevent import _requested_edges, _trajectory_payloads
    from nfit.symmetry import resolve_symmetry, symmetry_spec_from_config

    with zipfile.ZipFile(project) as archive:
        payload = json.loads(archive.read('project.json'))
    group = next(g for g in payload['data_groups'][0]['subgroups']
                 if g['name'] == 'YMn2 40 meV sample')
    config = group['metadata']['composite_binnings']['items'][0]['config']
    axes = config['axes']
    edges = _requested_edges(
        np.array([a['lower'] for a in axes]),
        np.array([a['upper'] for a in axes]),
        np.array([a['num_bins'] for a in axes]),
        np.array([a['step_size'] for a in axes]),
    )
    # Spread representative runs across the complete angular scan.
    runs = [group['datasets'][i] for i in np.linspace(0, len(group['datasets'])-1, 4, dtype=int)]
    detectors, transforms = _trajectory_payloads(
        SimpleNamespace(metadata=group['metadata']),
        [SimpleNamespace(metadata=r['metadata']) for r in runs],
        np.linalg.inv(np.array([a['vector'] for a in axes])),
        [op.matrix_hkl for op in resolve_symmetry(symmetry_spec_from_config(config['symmetry']))],
    )
    assert len(detectors) == 1
    _, theta, phi, solid = detectors[0]
    np.savez(destination, theta=theta, phi=phi, solid=solid,
             inverse=np.array([r[0] for r in transforms]),
             ei=np.array([r[1] for r in transforms]),
             limits=np.array([r[2] for r in transforms]),
             charge=np.array([r[3] for r in transforms]),
             **{f'edge{i}': edge for i, edge in enumerate(edges)})
    print(json.dumps({'host': socket.gethostname(), 'runs': len(runs),
                      'transforms': len(transforms), 'detectors': len(theta),
                      'shape': [len(e)-1 for e in edges]}), flush=True)


def prepare_synthetic(destination):
    """Reproduce the local medium-grid fixture; angles are synthetic."""
    rng = np.random.default_rng(47)
    theta = rng.uniform(.15, 2.4, 12000)
    phi = rng.uniform(-np.pi, np.pi, 12000)
    ub = np.array([
        [-.08957995796088149, -.09246997711819746, .0261999170762534],
        [.09452994551744022, -.09123995598517033, .0011599609645747946],
        [.017369958324288053, .019639892452625703, .12874006958665665],
    ])
    basis = np.array([[1, 1, 0], [1, -1, 0], [0, 0, 1]])
    matrices = []
    for angle in np.linspace(-np.pi, np.pi, 16):
        c, s = np.cos(angle), np.sin(angle)
        gonio = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        matrices.append(np.linalg.inv(basis).T @ np.linalg.inv(gonio @ (2*np.pi*ub)))
    edges = [np.linspace(-4.025, 4.025, 82), np.linspace(-4.025, 4.025, 82),
             np.linspace(-5.525, 5.525, 112), np.linspace(-.25, 40.25, 42)]
    np.savez(destination, theta=theta, phi=phi, solid=np.ones(len(theta)),
             inverse=np.array(matrices), ei=np.full(16, 40.126),
             limits=np.tile([-15., 38.], (16, 1)), charge=np.full(16, 33.32),
             **{f'edge{i}': edge for i, edge in enumerate(edges)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--mode', default='baseline')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--transforms', type=int, default=4)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--digest')
    args = parser.parse_args()
    if args.synthetic:
        prepare_synthetic(args.fixture)
        return
    if args.prepare:
        prepare(args.prepare, args.fixture)
        return
    import trajectory_candidates as candidates
    functions = {'baseline': candidates.run_baseline_dense,
                 'persistent': candidates.run_persistent_dense}
    if hasattr(candidates, 'run_axis0_slabs'):
        functions['slab'] = candidates.run_axis0_slabs
    fn = functions[args.mode]
    with np.load(args.fixture) as f:
        edges = [f[f'edge{i}'][::args.stride] for i in range(4)]
        shape = np.array([e.size-1 for e in edges])
        base = (f['theta'], f['phi'], f['solid'], f['inverse'], f['ei'],
                f['limits'], f['charge'], *edges, shape)
    batches = []
    for i in range(args.batches):
        start = i*args.transforms
        batches.append((*base[:3], *(a[start:start+args.transforms] for a in base[3:7]), *base[7:]))
    # Warm matching array signatures with a tiny output; exclude compilation.
    warm_edges = [np.linspace(e[0], e[-1], 3) for e in edges]
    warm = (*base[:3], *(a[:1] for a in base[3:7]), *warm_edges, np.array([2]*4))
    fn([warm], workers=args.workers)
    started = time.perf_counter()
    result = fn(batches, workers=args.workers)
    elapsed = time.perf_counter()-started
    # Bounded block digests enable equivalence checking without a second 3.7 GB result.
    digest = np.array([np.sum(result[i:i+100000], dtype=np.float64)
                       for i in range(0, result.size, 100000)])
    if args.digest:
        np.save(args.digest, digest)
    print(json.dumps({'host': socket.gethostname(), 'mode': args.mode,
                      'workers': args.workers, 'shape': shape.tolist(),
                      'bins': int(result.size), 'batches': args.batches,
                      'trajectories': args.batches*args.transforms*len(base[0]),
                      'seconds': elapsed, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                      'sum': float(np.sum(digest))}), flush=True)


if __name__ == '__main__':
    main()
