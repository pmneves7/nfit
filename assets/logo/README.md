# nfit logo sources

Preserve these design files for future editing:

- `nfit.ai`: Adobe Illustrator source (PDF-compatible file).
- `nfit-01.svg`: full wordmark, with the neutron sphere and “n” followed by “fit”.
- `nfit-02.svg`: neutron sphere with “n”, for the application icon.

Both SVGs have lettering converted to vector paths and require no installed
fonts. Each SVG contains an embedded 512 × 512 PNG sphere, rather than a
vector sphere. The icon sphere has slightly flattened top and left edges;
check the source image cropping before producing final application icons.

Keep the Illustrator source for editing and export text as outlines in future
SVG revisions. Check artboard bounds and small icon sizes before integrating
exports into the application or documentation. Preserve the design sources
when generating platform icon files.

The documentation wordmark is copied to `docs/_static/nfit-logo.svg`. The
application icon (`src/nfit/resources/nfit-icon.png`, 512 px) and documentation
favicon (`docs/_static/nfit-icon.png`, 64 px) are transparent square renders of
`nfit-02.svg`, centered with its aspect ratio preserved. Regenerate these
derivatives when updating the source designs. The documentation logo uses a
white backing so its black lettering remains readable in dark mode.
