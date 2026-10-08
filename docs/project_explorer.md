# Organize a project

The project explorer groups datasets, model components, fit history, analysis
recipes, and saved plots into workspaces. Expand a dataset collection to inspect
its contents. Large collections display run pages; these pages are views of the
underlying runs, not extra dataset groups.

Drag a run page to move all of its runs, or use the platform copy modifier to
copy them. Pages can stay collapsed. Selected pages and individual runs are
combined without duplicates. Drop onto a workspace or dataset collection to
add the runs there, or onto another run page to insert them at that page's
position. Dragging uses the same run selection as Copy.

## Selection and actions

Use Shift-click for a range or Ctrl/Command-click to select multiple compatible
items. Right-clicking an already selected row keeps the selection. Right-clicking
another row targets that row instead. Copy and Delete apply to the effective
selection, including all runs represented by selected run pages. Selecting both
a collection and its contents does not process the contents twice.

Use the context menu or the platform Copy and Paste shortcuts. Delete uses the
same selection as the context menu; the macOS Backspace key also deletes tree
items. When editing a name, Backspace edits the text.

## Copy and paste

Copies have independent editable settings and fresh identifiers. Immutable
numerical arrays can be shared without duplicating large datasets in memory.
Pasting repeatedly creates independent items rather than moving the originals.
Paste is available only at a compatible destination.

To duplicate a background or mask beside itself, select it, copy, and paste
while it remains selected. The copy stays in the same dataset or collection
and has independent settings and a unique name. Copied backgrounds keep their
source references.

| Copied items | Destination |
| --- | --- |
| Workspace | Project or another workspace, creating a sibling workspace |
| Datasets or run pages | Workspace or dataset collection |
| Dataset collections | Workspace or another dataset collection |
| Masks | Dataset, dataset collection, or an existing mask in either |
| Backgrounds | Dataset, dataset collection (including the root Datasets section), or an existing background |
| Model components | Workspace or Models |
| Fit history entries | Workspace or Fits |
| Analysis recipes | Workspace or Analyses |
| Saved plots | Workspace or Plots |

Copying a collection includes its contents. References between items included in
the copy follow their copied counterparts. A recipe or plot that depends on items
outside the copy requires those dependencies in the destination workspace.
Analysis-output rows represent results owned by their analysis and are not
independent clipboard objects. Copying an analysis alone copies its recipe and
clears its old results; run the pasted recipe to produce independent outputs.
Copying a complete workspace also copies and reconnects its existing outputs.
Copies of an Initial or Current fit state become saved history entries, so
later changes to the editable current state do not overwrite the pasted state.

## Independent project windows

The **Window** menu, between File and Help, offers **New project window** and
**Open project in new window…**. Each window runs in its own process with
independent edits, jobs, undo/history state and memory caches. Closing one window
does not close another. Performance preferences apply to each process, so several
active jobs share the machine's available resources. Save different projects to
different files; these windows do not coordinate simultaneous edits to one file.

Copy a tree selection in one window, then select a compatible destination and
paste in another. Configuration-only selections use the system clipboard.
Scientific selections use a lossless transfer archive in the project's temporary
workspace; an unsaved project without an existing data location asks where to
store it. Copying does not reduce sources or compute missing binnings. It writes
unsaved arrays and streams existing compressed cache artifacts without decoding
them. A large first copy may therefore take time proportional to its stored data.

Paste adopts independent archive ownership and leaves its data, reduced events
and histograms lazy. Pasted data remain usable after the source window closes or
its clipboard is replaced. Saving embeds owned arrays and reduced events into the
destination `.nfit` file; enable **File → Cache binnings** to retain its histograms
too. Original measurement and calibration files remain external references.
Current binnings transfer only when their source contents, masks, lattice and
numerical settings still match. Changed destination settings cause recomputation.

Copying a dataset collection also includes its linked measured-background source
collections/runs and retains inherited masks. A complete workspace is the simplest
way to transfer models, fits, analyses and plots with all their references.
Separately copied recipes can refer to counterparts already pasted into that
workspace. If the same source was pasted several times, the most recent copy is
used for such references. Dependencies absent from the destination are rejected.

Data viewers also have **Copy settings** and **Paste settings** buttons. Use
these to apply compatible slice ranges, zoom, color scaling, smoothing and
styling to a viewer in another project without copying its data. Axis names
determine which settings apply. Saved plot copies retain their full display
recipe and reconnect to copied sources. See [Data viewer](data_viewer.md).

**Window → Save copy/paste script…** saves the current copied selection as a
`.selection.nfit` archive alongside an editable Python script. Edit the destination
path and workspace in that script before running it. Keep the selection archive
with the script; the temporary clipboard archive is no longer needed.

## Delete and tree state

Deleting a collection removes its contents. Deleting the contents of a structural
section, such as Models, leaves the section itself in place. Fit-history
invariants still apply: the required initial state cannot be removed.
Deleting an analysis also removes the datasets explicitly derived from it.

Tree updates preserve the expanded and collapsed state of surviving collections
and pages, the current row, multiple selections, and the tree's scroll position.
Explicit navigation to a newly created or pasted item selects that item.
Removing datasets or groups does not remove the Models, Fits,
Analyses, or Plots sections. Missing inputs invalidate dependent recipes; they
do not prevent the rest of the project tree from being displayed.

Editing the selected item's settings preserves the details panel's tabs,
scroll position, and focused field when those controls still exist. Pending
layout updates do not return focus to an earlier field after you move elsewhere.

Selecting a dataset displays metadata and resident cached summaries; it does
not load source files, evaluate derived recipes, or rebin data to fill in a
summary. Exact fit-bin counts become available after the corresponding data
have been prepared. Use Load, Rebin now, a viewer, or a fit to prepare data.
Histogram counts are reused while their input data remain unchanged, and their
first calculation uses bounded working memory.
Rapid changes to settings share a queued refresh of open viewers. Manual
rebinning settings continue to defer computation until explicitly requested.
Point-list scale and wavelength fields apply when you press Enter or leave the
field, so intermediate numeric edits do not rebuild the controls.
Leaving a rebin axis field without changing its value does not invalidate the
data or rebuild the editor.

## Scripting

The GUI uses the GUI-independent `nfit.project_clipboard` service. Scripts can
use the same clipboard operations without constructing a Qt window:

```python
from nfit.project_clipboard import make_payload, paste_payload

payload = make_payload("dataset", [source_dataset])
result = paste_payload(
    payload,
    target_role="dataset_group",
    data_group=workspace,
    dataset_node=destination_collection,
)
copied_dataset = result.items[0]
```

`make_payload` snapshots the editable configuration. `paste_payload` validates
the destination and raises `ValueError` for incompatible operations. Use
`edit_project_file` when applying these changes to an existing `.nfit` archive so
the project state is reconciled and saved through the normal scripting workflow.

For transfer between processes or durable batch replay, use the public archive APIs:

```python
from nfit import export_project_items, import_project_items, save_project

export_project_items(source_project, "group", [source_workspace], "selection.nfit")
import_project_items(destination_project, "selection.nfit")
save_project(destination_project, "destination.nfit")
```

For selected runs, export with `role="dataset"` and `source_group=source_workspace`;
import with `target_role="datasets"`, `data_group=destination_workspace`, and
optionally `dataset_node=destination_collection`. The same validation, fresh IDs,
lazy ownership and cache-signature checks apply in scripts and the GUI.
