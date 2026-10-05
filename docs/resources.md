# Resource Manager

Click the **Resources** button to the right of the Project/System CPU and RAM readings.
The window lists flat rows that can be sorted by any column in either direction.
Use Shift-click for a range and Ctrl-click (Command-click on macOS) for separate
rows. The table lists source data, histogram stages and prepared views, reduced
events, evaluated models and overlays, and numerical fit/analysis results.
Inspecting or sorting rows never decodes an unloaded cube.

## Memory and disk columns

**RAM** is the distinct numerical storage held by that row. A shared array can
appear in several rows; the total numerical RAM counts it once. **Freed on
unload** estimates what removing that row and closing its affected viewers can
release. It can be zero when another cache still owns the arrays. Select related
rows together to release all those owners; individual freed amounts are not
additive. The details panel lists parents, sharing resources and viewers.

**Project cache** reports compressed storage in the saved `.nfit` archive;
**Temporary cache** reports session staging. Multiple rows can reference one
saved member. Reduced events stay on disk and are streamed in bounded chunks,
so there is no action to load an entire reduced-event table into RAM.

The summary also shows process resident RAM and outstanding operation
reservations. Resident RAM includes models, analysis, plotting libraries and
other allocations beyond the table's numerical payloads. Releasing arrays does
not guarantee that an allocator immediately returns their pages to the OS.

## Actions

- **Remove from memory** releases selected runtime owners. Affected viewers
  close; their current plot settings are retained for reopening in the same
  session. Saved plots remain unchanged. Source data can be unloaded only when
  their exact contents are recoverable from a source or project artifact.
  Unsaved replacement data is protected. Saved caches and recipes remain.
- **Load into memory** decodes saved caches or computes missing histogram
  recipes through the ordinary preparation APIs. Whole cubes remain resident
  for quick dataset switching and slider movement. Other open viewers reuse
  unchanged immutable prepared data, including scale and spectral conversions.
- **Delete cache from project** requires confirmation. It forgets the selected
  backing and marks that saved member for removal on the next successful save.
  It preserves currently loaded cubes. Deleted histogram caches are not
  silently regenerated during saving; explicitly loading their recipe makes
  them eligible for saving again. This action does not delete canonical source
  data, fit results, or analysis artifacts.

Actions are disabled while a project operation is active. A mixed selection must
support the chosen action on every row. Removing runtime data is distinct from
**File → Clear all caches in project**, which invalidates all computed caches.

## Budgets and temporary storage

The CPU and RAM controls edit the same machine-local limits as **Preferences →
Performance**. RAM is displayed in decimal GB (1 GB = 1,000,000,000 bytes), with
the scripting setting retained in MiB. Zero selects Auto. Changing a budget
does not evict loaded histograms.

Select an existing writable temporary directory to control future scientific
staging. The choice is saved in the project. Existing staging stays in place
until its owners release it; files are not moved during a directory change.
The temporary total covers live nfit-owned staging directories, not unrelated
files in the chosen directory. Atomic archive replacements stage beside their
save destination so that publication stays on the same filesystem.

Scientific staging is grouped under `.nfit-work` in the selected directory, or
beside the project or source file when no directory is selected. Each owner has
a private session folder containing the operation folders and a small ownership
marker with the hostname, process ID, and process creation time. Opening an empty
application creates no workspace. File-backed scientific data never falls back
to system temporary storage when its owner directory is unavailable.

A newly created `.nfit-work` container on POSIX is shared and sticky, like a shared scratch
directory: users can create their own entries but cannot remove another user's
entries. Each effective user ID has a separate private owner namespace, and owner
and session folders have `0700` permissions. Existing container permissions are
preserved. This keeps staging private while allowing collaborators to use the
same writable experiment directory.
On Windows, namespaces use the user's home identity and privacy follows the
directory access controls; POSIX permission bits are not used to judge them.

Successful saves adopt project-backed cache references. The corresponding
temporary files disappear after their last runtime owner or reader releases
them; copied dataset entries may still need them. Empty session and owner folders
are removed automatically. After a crash, inspect the remaining sessions before
cleanup. Automatic cleanup removes only confirmed dead processes on the current
host, checking process creation time to distinguish reused process IDs. Live
sessions, other hosts, invalid markers, and older flat `nfit-*` folders are
protected. Old folders without ownership markers require manual review.

An allocation that would exceed the configured budget or available system RAM
stops and offers Resource Manager. Unload objects or raise the budget, then retry.
Archive headers, resolved output grids, working buffers and concurrent
reservations are checked before managed allocations. This is an admission
policy, not an OS-enforced memory limit for third-party code.

Loading, decoding and saving use a guarded worker with progress and cooperative
cancellation. Cancelled atomic saves leave the previous project intact. Archive
failures are reported without clearing unsaved changes. Cached
histograms use standard compressed NPZ members within the project and unchanged
members are copied without recompression. Slice-only storage and slider caches
are not required: viewing continues to use full resident cubes.

## Scripting

The lifecycle service is available without Qt:

```python
from nfit.project_resources import ProjectResources
from nfit.data_workspace import set_project_temporary_directory

resources = ProjectResources(project)
for row in resources.snapshot().rows:
    print(row.key, row.state, row.ram_bytes, row.reclaimable_bytes)

# Use keys selected from the metadata-only snapshot.
resources.unload(selected_keys)
resources.load(saved_cache_keys)
resources.delete(cache_keys)  # removal is committed by the next project save
set_project_temporary_directory(project, "/path/to/experiment/work")
```

Inspect crash leftovers independently of a loaded project:

```python
from nfit.data_workspace import (
    inspect_temporary_workspaces, cleanup_temporary_workspaces,
)

owner = "/path/to/experiment/project.nfit"
for session in inspect_temporary_workspaces(owner, include_legacy=True):
    print(session.path, session.state, session.reason)

# Rechecks ownership and removes only dead same-host managed sessions.
removed = cleanup_temporary_workspaces(owner)
```

For a project that used a separate temporary directory, pass that directory as
`directory="/path/to/experiment/work"` to inspection and cleanup. These calls do
not relocate live staging or create a workspace.

Scripts with external consumers can supply `viewer_payloads` and
`release_viewers` callbacks to declare those references. Missing recipe loaders
can be supplied through `recipes`; ordinary scientific preparation remains
available through `dataset_for_slice_viewer` and `composite_dataset_entry`.
Use `nfit.resource_budget.reserve_memory` around extension-owned allocations.
