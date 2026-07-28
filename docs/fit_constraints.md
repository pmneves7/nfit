# Fit constraints

nfit fit constraints are hard relationships between model parameters. They are
stored in the project and fit-state snapshots, used by GUI and backend fits, and
included when a fit-generating script restores a saved state. Constraints are
therefore scientific configuration, not transient GUI state.

## Constraint editor

Select a workspace's top-level **Models** folder in the Project Explorer. The
**Fit constraints** table has one row per relationship:

- **Dependent parameter** is the parameter calculated or bounded by the row.
- **Relation** is exact equality (`=`), greater than or equal (`>=`), or less
  than or equal (`<=`). Numerical optimizers cannot enforce a meaningful strict
  `>` or `<`, so nfit uses inclusive boundaries.
- **Expression** is the right-hand side. Parameter references are qualified by
  model component and enclosed in backticks, for example `` `Signal.scale` ``.
- **Remove** deletes the relationship.

Use **Add constraint** to append a row. Start typing a backtick in the expression
editor to see parameter completion suggestions. **Check constraints** validates
the table without running a fit. Every interactive control has a tooltip with
the same rules.

The dependent parameter must:

- have **Fit** enabled,
- use **Global** sharing, and
- have blank Min and Max fields.

Parameters referenced on the right-hand side must also use global sharing, but
they may be fitted or fixed. A parameter may be the dependent parameter in only
one constraint. Renaming a model component through the project tree updates its
saved constraint references.

## Exact relationships

Choose `=` when one parameter is completely determined by other parameters or
constants. nfit removes the dependent parameter from the optimizer and evaluates
the expression for every model call.

| Desired relationship | Dependent parameter | Expression |
| --- | --- | --- |
| Equal parameters, $a=b$ | `A.a` | `` `B.b` `` |
| Fixed sum, $a+b=10$ | `A.a` | ``10 - `B.b` `` |
| Fixed product, $ab=10$ | `A.a` | ``10 / `B.b` `` |
| Fixed ratio, $a/b=2$ | `A.a` | ``2 * `B.b` `` |
| Positive square, $a=b^2$ | `A.a` | `` `B.b` ** 2 `` |

Expressions support numeric constants, parentheses, `+`, `-`, `*`, `/`, `%`,
and `**`. The allowed scalar functions are `abs`, `sqrt`, `exp`, `log`, `log10`,
`sin`, `cos`, and `tan`. Expressions are parsed by nfit's restricted arithmetic
evaluator; arbitrary Python, attribute access beyond parameter names, keyword
arguments, and imports are rejected.

Exact constraints may be chained. For example, `B.b = 2 * A.a` and
`C.c = B.b + 1` are evaluated in dependency order. Direct and indirect cycles
are rejected. Expressions must be finite at the starting parameter values and
throughout the fit; division by zero, invalid logarithms, and similar domain
errors stop the fit with a diagnostic rather than silently changing the model.

## Inequalities

Choose `>=` or `<=` for a hard ordering or one-sided threshold. The right-hand
side must currently be one qualified global parameter or one numeric constant:

| Desired relationship | Dependent parameter | Relation | Expression |
| --- | --- | --- | --- |
| $a \ge b$ | `A.a` | `>=` | `` `B.b` `` |
| $a \le 4$ | `A.a` | `<=` | `4` |

nfit enforces an inequality by reparameterization. For $a \ge b$, it fits a
nonnegative offset $\delta$ and evaluates $a=b+\delta$; for $a \le b$, it uses
$a=b-\delta$. The optimizer therefore cannot step outside the allowed region.
General arithmetic expressions on the right-hand side of inequalities are not
yet supported; use an exact helper relationship when the same condition can be
written with an additional model parameter.

## Reduced chi-squared

nfit reports

$$
\chi^2_\nu = \frac{\chi^2}{N-p},
$$

where $N$ is the number of included measured points and $p$ is the number of
independent optimizer variables.

An exact constraint removes its dependent parameter from $p$. An inequality
replaces the dependent parameter with an independently fitted offset, so it does
not reduce $p$. Fixed parameters are also excluded from $p$. Fit-result metadata
stores `n_points`, `n_variables`, and `degrees_of_freedom` alongside `chi2` and
`reduced_chi2`, making the calculation inspectable. If a model has no independent
variables, nfit uses all included points as the degrees of freedom.

## Backend configuration

The GUI writes plain dictionaries to `ModelComponentSpec.constraints`. The same
configuration can be created directly in a readable script:

```python
from nfit import (
    FitDatasetInput,
    ModelComponentSpec,
    compile_fit_problem,
    fit_problem_least_squares,
)

background = ModelComponentSpec(
    name="Background",
    type="constant_background",
    parameters={"constant": 1.0},
    fit_parameters={"constant": True},
)

signal = ModelComponentSpec(
    name="Signal",
    type="constant_background",
    parameters={"constant": 2.0},
    fit_parameters={"constant": True},
    constraints=[
        {
            "parameter": "constant",
            "op": "=",
            "expression": "10 - `Background.constant`",
        }
    ],
)

compiled = compile_fit_problem(
    [background, signal],
    [FitDatasetInput("scan", point_data)],
)
result = fit_problem_least_squares(compiled.problem)

print(result.params["Background.constant"])
print(result.params["Signal.constant"])
print(result.reduced_chi2)
```

An inequality uses `reference` instead of `expression`:

```python
signal.constraints = [
    {
        "parameter": "constant",
        "op": ">=",
        "reference": "Background.constant",
    }
]
```

The parameter named by `parameter` is local to the component containing the
constraint. References and expression variables use the full
`Component.parameter` name. This is the same serialized form saved in `.nfit`
projects and fit-history snapshots.

## Validation and diagnostics

The GUI's **Check constraints** action and fit compiler detect:

- unknown target or referenced parameters,
- duplicate constraints on one dependent parameter,
- dependent parameters that are fixed, bounded, or not globally shared,
- referenced parameters that are not globally shared,
- unsupported relations or expression syntax,
- non-finite starting expressions, and
- cyclic exact relationships.

Constraint errors abort the fit before optimization. The Project Explorer and
progress dialog remain usable so the relationship can be corrected and checked
again.
