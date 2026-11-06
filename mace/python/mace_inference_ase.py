"""Optional ASE adapter. ASE is not required by the core NumPy frontend."""
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from mace_inference import MH1


class MH1Calculator(Calculator):
    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(self, model, backend="auto", head=None, library=None, **kwargs):
        super().__init__(**kwargs)
        self.model = MH1(model, backend, head, library)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if not self.atoms.pbc.all():
            raise ValueError("MH1 requires three periodic axes")
        result = self.model.calculate(
            self.atoms.positions, self.atoms.numbers, self.atoms.cell.array
        )
        self.results = {key: result[key] for key in ("energy", "energies", "forces")}
        self.results["stress"] = full_3x3_to_voigt_6_stress(result["stress"])

    @property
    def diagnostics(self):
        return self.model.diagnostics

    def close(self):
        self.model.close()
