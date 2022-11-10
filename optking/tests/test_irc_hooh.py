#  IRC for HOOH from cis confirmation.
import psi4
import optking
import json
from .utils import utils

psi4.set_memory("2 GB")


def test_hooh_irc(check_iter):
    energy_5th_IRC_pt = -150.812913276783  # TEST
    h2o2 = psi4.geometry(
        """
      H     0.0000000000   0.9803530335  -0.8498671785
      O     0.0000000000   0.6988545188   0.0536419016
      O     0.0000000000  -0.6988545188   0.0536419016
      H     0.0000000000  -0.9803530335  -0.8498671785
    """
    )
    # Necessary since IRC will break C2h.
    h2o2.reset_point_group("c2")

    psi4.core.clean_options()

    psi4_options = {
        "basis": "dzp",
        "scf_type": "pk",
        "g_convergence": "gau_verytight",
        "opt_type": "irc",
        "geom_maxiter": 60,
        "full_hess_every": 0,
    }

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4("hf")
    print(json.dumps(json_output, indent=2))
    IRC = json_output["extras"]["irc_rxn_path"]

    print("%15s%15s%20s%15s" % ("Step Number", "Arc Distance", "Energy", "HOOH dihedral"))
    for step in IRC:
        print("%15d%15.5f%20.10f%15.5f" % (step["step_number"], step["arc_dist"], step["energy"], step["q"][5]))

    assert psi4.compare_values(energy_5th_IRC_pt, IRC[5]["energy"], 6, "Energy of 5th IRC point.")  # TEST
    utils.compare_iterations(json_output, 45, check_iter)
