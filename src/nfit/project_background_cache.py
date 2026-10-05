"""Persistence of background histograms owned by sample composite binnings."""

from . import project_composites as comp
from .analysis.artifacts import write_dataset_artifact
from .project_archive import ArchiveMember, binning_artifact_member, project_artifact_exists
from .project_resources import excluded_project_cache


def background_binning_artifacts(project, directory, format_version):
    """Save current owned backgrounds without preparing or loading unchanged cubes."""
    artifacts, entries = {}, []
    shared_members = {}
    for group_index, root in enumerate(project.data_groups):
        for scope in [root, *(comp._composite_scope(root, node) for node in root.iter_subgroups())]:
            node_id = scope.node.id if isinstance(scope, comp._CompositeScope) else None
            for binning in comp.data_group_composite_binnings(scope):
                config = binning['config']
                if not config.get('enabled'):
                    continue
                for index, background in enumerate(scope.backgrounds):
                    key = comp._composite_background_cache_key(scope, background, config)
                    signature = comp._composite_background_signature(scope, background, config)
                    cache = comp._COMPOSITE_DATA_CACHE
                    if not cache.has_signature(key, signature):
                        continue
                    owner = node_id or f'root-{group_index}'
                    cache_id = f'background-{owner}-{binning["id"]}-{index}'
                    member = binning_artifact_member(cache_id)
                    if excluded_project_cache(project, member):
                        continue
                    backing = cache.project_backing(key, signature)
                    if backing is not None and excluded_project_cache(project, backing[1]):
                        continue
                    if signature in shared_members:
                        member = shared_members[signature]
                        if excluded_project_cache(project, member):
                            continue
                    elif backing is not None:
                        artifacts[member] = ArchiveMember(*backing)
                    else:
                        path = directory / f'{cache_id}.npz'
                        write_dataset_artifact(cache.get(key)[1], path)
                        artifacts[member] = path
                    shared_members[signature] = member
                    entries.append(dict(
                        type='composite_background', format_version=format_version,
                        group_index=group_index, node_id=node_id,
                        binning_id=binning['id'], background_index=index,
                        signature=signature, member=member,
                    ))
    return artifacts, entries


def restore_background_binning_backing(project, path, entries, *, lazy):
    """Validate source/grid signatures and register saved histograms lazily."""
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('type') != 'composite_background':
            continue
        try:
            root = project.data_groups[int(entry['group_index'])]
            node_id = entry.get('node_id')
            node = None if node_id is None else next(n for n in root.iter_subgroups() if n.id == node_id)
            scope = comp._composite_scope(root, node)
            config = comp.data_group_composite_config_by_id(scope, entry['binning_id'])
            background = scope.backgrounds[int(entry['background_index'])]
            signature = comp._composite_background_signature(scope, background, config)
            if signature != entry['signature'] or not project_artifact_exists(path, entry['member']):
                continue
            comp._COMPOSITE_DATA_CACHE.set_project_backing(
                comp._composite_background_cache_key(scope, background, config),
                signature=signature, project_path=path, member=entry['member'], lazy=lazy,
            )
        except (IndexError, KeyError, OSError, StopIteration, TypeError, ValueError):
            continue
