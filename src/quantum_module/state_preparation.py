"""
Quantum State Preparation for Brain Tumor Segmentation Samples

This module prepares quantum states from the CVAE-generated segmentation samples.
We use amplitude encoding to represent the superposition of all possible segmentations.

Key Concept:
    Classical: Store N samples in memory
    Quantum: Encode N samples in log2(N) qubits as superposition

    |ψ⟩ = 1/√N ∑_{i=0}^{N-1} |i⟩

    Where each |i⟩ corresponds to one segmentation sample
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from typing import List, Tuple, Callable
import math


class SegmentationStatePreparation:
    """
    Prepares quantum states representing segmentation samples

    The state encoding:
    - Each segmentation sample gets a unique index i ∈ [0, N-1]
    - We need ceil(log2(N)) qubits to represent N samples
    - Initial state: |ψ⟩ = 1/√N ∑_i |i⟩ (uniform superposition)
    """

    def __init__(self, num_samples: int):
        """
        Args:
            num_samples: Number of segmentation samples to encode
        """
        self.num_samples = num_samples
        self.num_qubits = math.ceil(math.log2(num_samples))
        self.padded_samples = 2 ** self.num_qubits  # Round up to power of 2

        print(f"State Preparation Configuration:")
        print(f"  Samples: {num_samples}")
        print(f"  Qubits needed: {self.num_qubits}")
        print(f"  Padded size: {self.padded_samples}")

    def create_uniform_superposition(self) -> QuantumCircuit:
        """
        Create uniform superposition over all sample indices

        Returns:
            |psi> = 1/sqrt(N) sum_{i=0}^{N-1} |i>
        """
        qc = QuantumCircuit(self.num_qubits, name='State Prep')

        # Apply Hadamard to all qubits for uniform superposition
        qc.h(range(self.num_qubits))

        # If we padded, we need to zero out invalid states
        # For simplicity in this proof-of-concept, we accept the padding
        # In production, you'd use amplitude amplification to handle this

        return qc

    def create_weighted_superposition(self, weights: np.ndarray) -> QuantumCircuit:
        """
        Create weighted superposition based on sample probabilities

        This allows encoding prior knowledge or importance weights

        Args:
            weights: Probability weights for each sample (must sum to 1)

        Returns:
            |psi> = sum_{i=0}^{N-1} sqrt(w_i) |i>
        """
        assert len(weights) == self.num_samples, \
            f"Weights length {len(weights)} != num_samples {self.num_samples}"
        assert np.isclose(weights.sum(), 1.0), \
            f"Weights must sum to 1, got {weights.sum()}"

        # Pad weights to power of 2
        padded_weights = np.zeros(self.padded_samples)
        padded_weights[:self.num_samples] = weights

        # Normalize after padding
        padded_weights = padded_weights / np.linalg.norm(padded_weights)

        # Create state preparation circuit
        qc = QuantumCircuit(self.num_qubits, name='Weighted State Prep')

        # Use Qiskit's StatePreparation
        state_prep = StatePreparation(padded_weights)
        qc.append(state_prep, range(self.num_qubits))

        return qc


class SegmentationDatabase:
    """
    Database of segmentation samples with metadata

    This stores the actual segmentation data and provides
    the classical function that the quantum oracle will encode
    """

    def __init__(self, segmentations: np.ndarray):
        """
        Args:
            segmentations: Array of segmentation predictions
                          Shape: (num_samples, D, H, W)
        """
        self.segmentations = segmentations
        self.num_samples = len(segmentations)

        print(f"\nSegmentation Database:")
        print(f"  Samples: {self.num_samples}")
        print(f"  Shape per sample: {segmentations.shape[1:]}")

    def get_segmentation(self, index: int) -> np.ndarray:
        """Get segmentation at given index"""
        return self.segmentations[index]

    def evaluate_property(
        self,
        index: int,
        property_fn: Callable[[np.ndarray], bool]
    ) -> bool:
        """
        Evaluate a property function on segmentation at index

        Args:
            index: Sample index
            property_fn: Function that takes segmentation and returns bool

        Returns:
            True if segmentation satisfies property
        """
        seg = self.get_segmentation(index)
        return property_fn(seg)

    def count_satisfying(self, property_fn: Callable[[np.ndarray], bool]) -> int:
        """
        Classical counting of samples satisfying property

        This is the O(N) classical approach that QAE will beat!

        Args:
            property_fn: Boolean function to evaluate

        Returns:
            Count of samples satisfying property
        """
        count = 0
        for i in range(self.num_samples):
            if self.evaluate_property(i, property_fn):
                count += 1
        return count

    def estimate_probability_classical(
        self,
        property_fn: Callable[[np.ndarray], bool]
    ) -> float:
        """
        Classical probability estimation

        Time complexity: O(N)

        Returns:
            P(property is true) = count / num_samples
        """
        count = self.count_satisfying(property_fn)
        return count / self.num_samples


def create_sample_database(num_samples: int = 16) -> Tuple[SegmentationDatabase, QuantumCircuit]:
    """
    Create a sample database for testing

    Args:
        num_samples: Number of dummy samples to create

    Returns:
        database: SegmentationDatabase with dummy data
        state_prep_circuit: Quantum circuit for state preparation
    """
    # Create dummy segmentations (simplified 3D volumes)
    # In practice, these come from the CVAE sampler
    segmentations = np.random.randint(0, 4, size=(num_samples, 32, 32, 32))

    # Create database
    database = SegmentationDatabase(segmentations)

    # Create state preparation
    state_prep = SegmentationStatePreparation(num_samples)
    circuit = state_prep.create_uniform_superposition()

    return database, circuit


if __name__ == "__main__":
    """Demo of quantum state preparation"""
    print("="*60)
    print("Quantum State Preparation Demo")
    print("="*60)

    # Example 1: Uniform superposition
    print("\n1. Creating uniform superposition over 16 samples")
    state_prep = SegmentationStatePreparation(num_samples=16)
    circuit = state_prep.create_uniform_superposition()

    print(f"\nQuantum Circuit created with {circuit.num_qubits} qubits and {circuit.depth()} depth")

    print(f"\nThis creates the state:")
    print(f"  |psi> = 1/sqrt(16) (|0> + |1> + ... + |15>)")
    print(f"  Using {state_prep.num_qubits} qubits")

    # Example 2: Weighted superposition
    print("\n" + "="*60)
    print("\n2. Creating weighted superposition")
    weights = np.random.random(16)
    weights = weights / weights.sum()  # Normalize

    weighted_circuit = state_prep.create_weighted_superposition(weights)
    print(f"\nWeighted state with non-uniform probabilities")
    print(f"  Top 3 weights: {sorted(weights, reverse=True)[:3]}")

    # Example 3: Sample database
    print("\n" + "="*60)
    print("\n3. Creating sample database")
    database, prep_circuit = create_sample_database(num_samples=16)

    # Define a sample property (e.g., has tumor in upper half)
    def has_tumor_in_upper_half(seg: np.ndarray) -> bool:
        """Check if tumor exists in upper half of volume"""
        upper_half = seg[:seg.shape[0]//2]
        return (upper_half > 0).any()

    # Classical counting
    print("\n4. Classical probability estimation (O(N) approach)")
    classical_prob = database.estimate_probability_classical(has_tumor_in_upper_half)
    print(f"  P(tumor in upper half) = {classical_prob:.3f}")
    print(f"  Required {database.num_samples} evaluations")

    print("\n" + "="*60)
    print("\n[SUCCESS] State preparation demo complete!")
    print("\nKey Insights:")
    print("  - Classical: Store N samples -> O(N) memory")
    print("  - Quantum: Encode N samples in log2(N) qubits")
    print("  - Classical counting: O(N) evaluations")
    print("  - Quantum QAE: O(sqrt(N)) queries -> Quadratic speedup!")
    print("\nNext: Build the oracle and QAE algorithm")
