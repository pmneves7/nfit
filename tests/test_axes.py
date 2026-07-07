from metallix import infer_axis_role


def test_infer_axis_role_from_reduced_data_labels_and_units():
    assert infer_axis_role("H", units="r.l.u.") == "h"
    assert infer_axis_role("K", units="r.l.u.") == "k"
    assert infer_axis_role("L", units="r.l.u.") == "l"
    assert infer_axis_role("|Q|", units="Angstrom^-1") == "q_modulus"
    assert infer_axis_role("DeltaE", units="meV") == "energy_transfer"
    assert infer_axis_role("signal") == "intensity"
    assert infer_axis_role("sigma") == "uncertainty"
    assert infer_axis_role("[H,H,0]", units="r.l.u.") == "momentum_projection"
