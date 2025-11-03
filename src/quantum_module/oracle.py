"""
Quantum Oracle for Clinical Queries

This module defines oracles that mark quantum states corresponding to
segmentations satisfying clinical properties (e.g., multifocality).

Oracle Definition:
    O |i> = (-1)^f(i) |i>

    Where f(i) = 1 if segmentation i satisfies the property
               = 0 otherwise

The oracle acts as a phase flip on "good" states, which is essential for
Grover's algorithm and Quantum Amplitude Estimation.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from scipy import ndimage
from typing import Callable, List
import math

from state_preparation import SegmentationDatabase


class MultifocalityOracle:
    """
    Oracle for detecting multifocal tumors

    A tumor is multifocal if it has multiple disconnected components.
    This is a clinically important question that impacts treatment planning.

    The oracle marks all segmentation states where the tumor is multifocal.
    """

    def __init__(
        self,
        database: SegmentationDatabase,
        connectivity: int = 1
    ):
        """
        Args:
            database: Database of segmentation samples
            connectivity: Connectivity for connected components (1, 2, or 3)
        """
        self.database = database
        self.connectivity = connectivity
        self.num_samples = database.num_samples
        self.num_qubits = math.ceil(math.log2(self.num_samples))

        # Precompute which samples are multifocal (classical preprocessing)
        self.multifocal_indices = self._find_multifocal_samples()

        print(f"Multifocality Oracle:")
        print(f"  Samples: {self.num_samples}")
        print(f"  Multifocal samples: {len(self.multifocal_indices)}")
        print(f"  True probability: {len(self.multifocal_indices)/self.num_samples:.3f}")

    def _is_multifocal(self, segmentation: np.ndarray) -> bool:
        """
        Check if a segmentation is multifocal

        Args:
            segmentation: 3D array of class labels

        Returns:
            True if multifocal (>1 connected component)
        """
        # Create binary tumor mask (any non-background class)
        tumor_mask = segmentation > 0

        # Find connected components
        structure = ndimage.generate_binary_structure(3, self.connectivity)
        labeled, num_components = ndimage.label(tumor_mask, structure=structure)

        return num_components > 1

    def _find_multifocal_samples(self) -> List[int]:
        """
        Precompute which samples are multifocal

        Returns:
            List of indices of multifocal samples
        """
        multifocal = []
        for i in range(self.num_samples):
            seg = self.database.get_segmentation(i)
            if self._is_multifocal(seg):
                multifocal.append(i)
        return multifocal

    def create_phase_oracle(self) -> QuantumCircuit:
        """
        Create the phase oracle circuit

        The oracle flips the phase of all multifocal states:
            O |i> = (-1)^{is_multifocal(i)} |i>

        Implementation:
            For each multifocal index i, apply a multi-controlled Z gate
            that activates when the input register equals i

        Returns:
            Quantum circuit implementing the oracle
        """
        qc = QuantumCircuit(self.num_qubits, name='Multifocal Oracle')

        # For each multifocal sample, add a phase flip
        for idx in self.multifocal_indices:
            # Convert index to binary
            binary = format(idx, f'0{self.num_qubits}b')

            # Apply X gates to qubits that should be 0
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

            # Multi-controlled Z (flips phase if all qubits are 1)
            if self.num_qubits == 1:
                qc.z(0)
            elif self.num_qubits == 2:
                qc.cz(0, 1)
            else:
                # Multi-controlled Z using H-MCX-H pattern
                qc.h(self.num_qubits - 1)
                qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                qc.h(self.num_qubits - 1)

            # Undo the X gates
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

        return qc

    def create_comparison_oracle(self, ancilla_qubit: int) -> QuantumCircuit:
        """
        Create oracle that writes result to ancilla qubit

        This version uses an ancilla qubit to store whether the state
        is multifocal: |i>|0> -> |i>|f(i)>

        Args:
            ancilla_qubit: Index of ancilla qubit

        Returns:
            Quantum circuit
        """
        qc = QuantumCircuit(self.num_qubits + 1, name='Multifocal Oracle (Ancilla)')

        # For each multifocal sample, flip the ancilla
        for idx in self.multifocal_indices:
            binary = format(idx, f'0{self.num_qubits}b')

            # Apply X to qubits that should be 0
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

            # Controlled-X on ancilla (flip ancilla if all qubits match)
            qc.mcx(list(range(self.num_qubits)), ancilla_qubit)

            # Undo X gates
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

        return qc


class GenericOracle:
    """
    Generic oracle for any boolean property function

    This allows creating oracles for arbitrary clinical queries beyond
    just multifocality (e.g., tumor volume, location, etc.)
    """

    def __init__(
        self,
        database: SegmentationDatabase,
        property_fn: Callable[[np.ndarray], bool],
        name: str = "Generic Oracle"
    ):
        """
        Args:
            database: Database of segmentations
            property_fn: Function that takes segmentation and returns bool
            name: Name for the oracle
        """
        self.database = database
        self.property_fn = property_fn
        self.name = name
        self.num_samples = database.num_samples
        self.num_qubits = math.ceil(math.log2(self.num_samples))

        # Precompute satisfying samples
        self.satisfying_indices = self._find_satisfying_samples()

        print(f"{name}:")
        print(f"  Samples: {self.num_samples}")
        print(f"  Satisfying samples: {len(self.satisfying_indices)}")
        print(f"  True probability: {len(self.satisfying_indices)/self.num_samples:.3f}")

    def _find_satisfying_samples(self) -> List[int]:
        """Find all samples satisfying the property"""
        satisfying = []
        for i in range(self.num_samples):
            seg = self.database.get_segmentation(i)
            if self.property_fn(seg):
                satisfying.append(i)
        return satisfying

    def create_phase_oracle(self) -> QuantumCircuit:
        """Create phase oracle for this property"""
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        for idx in self.satisfying_indices:
            binary = format(idx, f'0{self.num_qubits}b')

            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

            if self.num_qubits == 1:
                qc.z(0)
            elif self.num_qubits == 2:
                qc.cz(0, 1)
            else:
                # Multi-controlled Z using H-MCX-H pattern
                qc.h(self.num_qubits - 1)
                qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                qc.h(self.num_qubits - 1)

            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(i)

        return qc


if __name__ == "__main__":
    """Demo of oracle construction"""
    from state_preparation import create_sample_database

    print("="*60)
    print("Quantum Oracle Demo")
    print("="*60)

    # Create sample database
    print("\n1. Creating sample database...")
    database, _ = create_sample_database(num_samples=16)

    # Create multifocality oracle
    print("\n2. Creating multifocality oracle...")
    oracle = MultifocalityOracle(database, connectivity=1)

    # Build oracle circuit
    print("\n3. Building oracle circuit...")
    oracle_circuit = oracle.create_phase_oracle()
    print(f"  Oracle circuit depth: {oracle_circuit.depth()}")
    print(f"  Oracle circuit gates: {oracle_circuit.size()}")

    # Test with specific property
    print("\n4. Creating custom oracle (large tumor)...")
    def has_large_tumor(seg: np.ndarray) -> bool:
        """Check if tumor volume > threshold"""
        tumor_voxels = (seg > 0).sum()
        total_voxels = seg.size
        return (tumor_voxels / total_voxels) > 0.1

    custom_oracle = GenericOracle(database, has_large_tumor, "Large Tumor Oracle")
    custom_circuit = custom_oracle.create_phase_oracle()

    print(f"  Custom oracle depth: {custom_circuit.depth()}")

    # Demonstrate oracle action
    print("\n5. Oracle marks these states:")
    print(f"  Multifocal indices: {oracle.multifocal_indices}")
    print(f"  These states get phase flip: -1")
    print(f"  All other states remain unchanged")

    print("\n" + "="*60)
    print("\n[SUCCESS] Oracle demo complete!")
    print("\nKey Points:")
    print("  - Oracle encodes classical boolean function")
    print("  - Phase oracle: O|i> = (-1)^f(i) |i>")
    print("  - Works for any property (multifocal, volume, location, etc.)")
    print("  - Essential for QAE to estimate probability")
    print("\nNext: Implement Quantum Amplitude Estimation algorithm")
