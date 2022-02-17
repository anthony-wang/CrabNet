from pymatgen.core import Composition


# %%
def guess_oxidation(form):
    comp = Composition(form)
    comp = comp.add_charges_from_oxi_state_guesses()
    out_dict = comp.oxi_state_guesses()
    if len(out_dict) > 1:
        return out_dict[0]
    return out_dict

def get_contents(x, form):
    if isinstance(x, tuple):
        out = x[0]
    if isinstance(x, dict):
        out = x
    if isinstance(x, list) and len(x) == 0:
        out = None
    if x is None:
        out = x
    return out

def find_oxidations(atom, dictionary):
    n_total = 0
    n_guesses = 0
    n_atoms = 0
    atom_states = []

    for k, v in dictionary.items():
        if v is None:
            atom_states.append(None)
        elif atom in v.keys():
            if len(v) > 0:
                n_atoms+=1
                atom_states.append(v[atom])
            n_guesses += 1
        else:
            atom_states.append(None)
        n_total += 1
    return n_total, n_guesses, n_atoms, atom_states

def get_ionic_or_nonionic(dictionary):
    res = []
    for v in dictionary.values():
        if v is None:
            # no oxidation recorded
            res.append(False)
        elif len(v) > 0:
            # oxidation recorded
            res.append(True)
    return res