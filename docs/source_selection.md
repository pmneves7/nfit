# Selecting numbered sources

Use **Dataset importing** in the workspace to enter a data directory, file
prefix, file suffix, number padding, and run expression. **Preview runs** expands
that expression without importing data. The preview shows grouped appearances,
missing files, and repeated physical sources. **Inspect acquisition metadata**
optionally reads available instrument, incident-energy, time-zero and temperature
headers without loading event arrays. Large previews display the first 1,000
rows and report totals for the entire selection.

**Import datasets** creates one ordinary nfit dataset group containing the unique
selected sources. Repeating a number does not add an independent measurement or
change its weight. Enable **Preserve expression groups as subfolders** to retain
expression groups as ordinary dataset subfolders. There is no second project
grouping system. **Add files** remains available for irregular filenames.

The saved group records the naming pattern, original expression, resolved paths,
source appearances, missing-file policy, and physical source identities. Native
DGS, CORELLI and MDEvent groups retain their normal
[reduction recipes](reduction_recipes.md), with one logical run per acquisition
or MDEvent experiment. File contents remain lazy external references.

## Expressions

Ranges include their endpoints; a stride can omit an endpoint it does not reach.
The parser limits expansion to 100,000 appearances and rejects invalid or
oversized expressions before importing anything.

| Expression | Meaning when preserving expression groups |
| --- | --- |
| `10,12,15` | Separate groups |
| `10:15` | One group per run |
| `10:::19` or `10:3:19` | One group per run, stride three |
| `10+12` | Both runs in one group |
| `10>15` | Inclusive range in one group |
| `10>>>19` | Range in one group, stride three |
| `10'12` | Advance the second grouping index |
| `10;15` | Range through the second grouping index; `;;;` uses stride three |
| `10(n;s1;s2;r)` | Sum all repeated blocks in one group |
| `10[n;s1;s2;r]` | One group containing each block |
| `10{n;s1;s2;r}` | One group per source through the first index |
| `10/n;s1;s2;r/` | One group per source through the second index |
| `10\|3\|` | Three groups referring to the same source |
| `10!3!` | Repeat through the second index |
| `0` or `0\|5\|` | One or five empty groups |
| `x` | Clear the selection, without deleting project data |

For block expressions, `n` is the number of sources per block, `s1` the number of
files skipped between sources within a block, `s2` the number skipped between
blocks, and `r` the repetition count. Omitted parameters default to `n;0;0;1`.
The two indices specify expression grouping only; retained groups use nfit's
existing dataset folders. Separate ranges with commas: `start:stride:end` takes
precedence over a chained range.

## Repeated sources and uncertainty

Flat import uses unique physical membership, including repeats written with `+`.
With preserved subfolders, repeating a raw DGS or MDEvent source within one group
applies its multiplicity as a coaddition coefficient. For multiplicity $m$, the
count numerator $C$ and exposure $N$ scale by $m$, while numerator variance $V_C$
scales by $m^2$. The estimate $C/N$ and its counting uncertainty are unchanged:
reusing one acquisition cannot create more counting precision. Exposure here is
the pipeline's normalization denominator, not necessarily elapsed time.

Repeated sources in different subfolders retain one physical identity. They can
be inspected separately. Combining those aliases raises a source-replay request
because the current composite path does not represent their shared uncertainty.
Separately evaluated fit datasets also reject shared acquisition identities;
uncertainty from a reused source cannot be treated as independent residual blocks.
This check is conservative at acquisition level: disjoint cuts require a source
representation that proves their independence. It does not infer independence
from their display coordinates.
Repeated CORELLI or ordinary sources within a summed group are also rejected
until their adapters support that coaddition. Unique runs retain the existing
instrument reduction and combination rules.

Empty subfolders are disabled and contain no observations. Missing files prevent
GUI import. Scripts can explicitly choose `missing="skip"`; skipped appearances
remain in provenance and empty retained folders remain disabled. One expression
group must use one native reduction format, or ordinary registered importers;
use separate dataset groups for different reduction families.

## Scripting

The GUI and scripts use the same public resolver and atomic import service:

```python
from nfit import DataGroup, SourceSelection, resolve_source_selection
from nfit import import_source_selection, source_selection_script

selection = SourceSelection(
    directory="/path/to/data", prefix="SEQ_", suffix=".nxs.h5",
    expression="392985:393469,393500:393631", padding=0,
)
preview = resolve_source_selection(selection, inspect_metadata=True)
workspace = DataGroup("NiO")
collection = import_source_selection(workspace, preview)
# Optional: preserve_groups=True creates ordinary dataset subfolders.
script = source_selection_script(selection, name="NiO runs")
```

**Copy source import script** exports editable selection and import settings.
Executing it resolves the current expression against the directory. By contrast,
a saved reduction-recipe replay uses its recorded resolved membership. Native event imports retain the existing companion-file rule: when both
calibration paths are omitted, a sole sibling file named `van*` supplies the
normalization and mask. The resolved paths are saved and visible in the reduction
panel, where they can be changed or cleared. Explicit calibration paths in scripts
replace that discovery.

Native
imports inspect run metadata but do not reduce event arrays. Registered importers
retain their existing lazy or eager loading behavior. A failed selection import
leaves the destination group unchanged.
