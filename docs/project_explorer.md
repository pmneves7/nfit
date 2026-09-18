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

| Copied items | Destination |
| --- | --- |
| Workspace | Project or another workspace, creating a sibling workspace |
| Datasets or run pages | Workspace or dataset collection |
| Dataset collections | Workspace or another dataset collection |
| Masks | Dataset or dataset collection |
| Backgrounds | Dataset or dataset collection |
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

## Delete and tree state

Deleting a collection removes its contents. Deleting the contents of a structural
section, such as Models, leaves the section itself in place. Fit-history
invariants still apply: the required initial state cannot be removed.
Deleting an analysis also removes the datasets explicitly derived from it.

Tree updates preserve the expanded and collapsed state of surviving collections
and pages. Removing datasets or groups does not remove the Models, Fits,
Analyses, or Plots sections. Missing inputs invalidate dependent recipes; they
do not prevent the rest of the project tree from being displayed.

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
