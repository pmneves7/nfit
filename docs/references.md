# References and software influences

nfit is an independent implementation, but it benefits from ideas established
by the neutron-scattering and scientific-software community. The citations
below acknowledge those influences; they do not imply API compatibility or
endorsement.

## Software

- **Mantid** established many of the NeXus, multidimensional-workspace,
  direct-geometry reduction, and visualization conventions used throughout the
  neutron community. See O. Arnold *et al.*, *Nucl. Instrum. Methods Phys. Res.
  A* **764**, 156–166 (2014),
  [doi:10.1016/j.nima.2014.07.029](https://doi.org/10.1016/j.nima.2014.07.029).
- **SHIVER** (Spectroscopy HIstogram Visualizer for Event Reduction) informed
  nfit's direct-geometry, multiple-orientation, HKLE histogramming workflow.
  See the [SHIVER documentation](https://shiver.readthedocs.io/).
- **Horace** is an important precedent for exploring, transforming, simulating,
  and fitting four-dimensional time-of-flight neutron data. See R. A. Ewings
  *et al.*, *Nucl. Instrum. Methods Phys. Res. A* **834**, 132–142 (2016),
  [doi:10.1016/j.nima.2016.07.036](https://doi.org/10.1016/j.nima.2016.07.036).
- **DAVE** demonstrated the value of an integrated environment for neutron-data
  reduction, visualization, and analysis. See R. T. Azuah *et al.*, *J. Res.
  Natl. Inst. Stand. Technol.* **114**, 341–358 (2009),
  [doi:10.6028/jres.114.025](https://doi.org/10.6028/jres.114.025).
- **SasView** influenced the model-driven fitting interface and reusable
  scientific-GUI patterns. nfit's `NDRebin` implementation is adapted from
  SasView's `sasdata` package; the required attribution and license text are in
  [Third-party licenses](https://github.com/pmneves7/nfit/blob/main/THIRD_PARTY_LICENSES.md).
  Cite the exact SasView
  release used in an analysis as described on the
  [SasView citation page](https://www.sasview.org/cite/).
- **Spinteract** is a close conceptual precedent for refining
  symmetry-constrained magnetic interactions against diffuse-scattering data.
  See J. A. M. Paddison, *J. Phys.: Condens. Matter* **35**, 495802 (2023),
  [doi:10.1088/1361-648X/acf261](https://doi.org/10.1088/1361-648X/acf261).
- **SpinW** and **Sunny.jl** informed the presentation of crystal structures,
  symmetry-related exchange bonds, and spin-Hamiltonian models. See S. Tóth
  and B. Lake, *J. Phys.: Condens. Matter* **27**, 166002 (2015),
  [doi:10.1088/0953-8984/27/16/166002](https://doi.org/10.1088/0953-8984/27/16/166002);
  and D. Dahlbom *et al.*, “Sunny.jl: A Julia Package for Spin Dynamics”
  (2025), [arXiv:2501.13095](https://doi.org/10.48550/arXiv.2501.13095).

Use this list to credit intellectual and interface influences. Software
actually used to reduce, simulate, or fit published data should also be cited
at the version used.

## Physics and numerical references

The following sources underlie nfit's conventions and model documentation:

- G. L. Squires, *Introduction to the Theory of Thermal Neutron Scattering*,
  3rd ed. (Cambridge University Press, 2012),
  [doi:10.1017/CBO9781139107808](https://doi.org/10.1017/CBO9781139107808).
  Chapter 7 fixes the magnetic correlation-function and cross-section
  convention used by nfit.
- S. W. Lovesey, *Theory of Neutron Scattering from Condensed Matter*,
  Vols. 1–2 (Clarendon Press, 1984).
- I. A. Zaliznyak and S.-H. Lee, “Magnetic Neutron Scattering,” in
  *Modern Techniques for Characterizing Magnetic Materials* (Springer, 2005),
  [doi:10.1007/0-387-23395-4_1](https://doi.org/10.1007/0-387-23395-4_1).
- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*
  (Springer, 1985),
  [doi:10.1007/978-3-642-82499-9](https://doi.org/10.1007/978-3-642-82499-9).
- P. J. Brown, “Magnetic form factors,” *International Tables for
  Crystallography*, Vol. C, §4.4.5; see the
  [ILL magnetic form-factor tables](https://www.ill.eu/sites/ccsl/ffacts/).
- G. J. Feldman and R. D. Cousins, *Phys. Rev. D* **57**, 3873–3889 (1998),
  [doi:10.1103/PhysRevD.57.3873](https://doi.org/10.1103/PhysRevD.57.3873),
  for confidence intervals used by the measured-zero uncertainty convention.

Model-specific papers are cited beside the equations on
[Spin-fluctuation models](spin_fluctuation_models.md) and
[Theory: sum rules and self-consistency](theory_notes.md).
