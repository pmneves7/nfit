# nfit logo sources

Preserve these design files for future editing:

- `nfit.ai`: Adobe Illustrator source (PDF-compatible file).
- `nfit-01.svg`: full wordmark, with the neutron sphere and “n” followed by “fit”.
- `nfit-02.svg`: neutron sphere with “n”, for the application icon.

Both SVGs use vector paths for the lettering and a radial-gradient vector
sphere, so they require no installed fonts and remain sharp at any size.

Keep the Illustrator source for editing and export text as outlines in future
SVG revisions. Check artboard bounds and small icon sizes before integrating
exports into the application or documentation. Preserve the design sources
when generating platform icon files. The PNG icon derivatives remain necessary
because operating-system application icons use raster representations at fixed
sizes.

The documentation wordmark is copied to `docs/_static/nfit-logo.svg`. The
application icon (`src/nfit/resources/nfit-icon.png`, 512 px) and documentation
favicon (`docs/_static/nfit-icon.png`, 64 px) are transparent square renders of
`nfit-02.svg`, centered with its aspect ratio preserved. Regenerate these
derivatives when updating the source designs. The dark-mode copy
(`docs/_static/nfit-logo-dark.svg`) changes only the “fit” paths to white;
the sphere and black “n” remain unchanged. Both themes use transparent backgrounds.
