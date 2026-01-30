from typing import Optional,List
from src.cluster import Cluster
import numpy as np
from numpy.typing import NDArray
from src.trajectory_utils import (
    stationary_distribution,
    time_reversed_transition_matrix,
    count_transitions,
    metastability,
    entropy_rate,
)
from src.embedding_base import EmbeddingBase

def _sort_idx(vals: np.ndarray) -> np.ndarray:
    # sort by increasing Re(λ), break ties by |λ|
    return np.lexsort((np.abs(vals), np.real(vals)))

class StochasticMatrix:
    def __init__(self, P: NDArray[np.float64]) -> None:
        self.P = P
        self.pi: Optional[np.ndarray] = stationary_distribution(P)
        self.Pr: Optional[np.ndarray] = None
        self.slow_mode: Optional[np.ndarray] = None
        self.tr_slow_mode: Optional[np.ndarray] = None

    def reversibilized_matrix(self) -> NDArray[np.float64]:
        if self.pi is None or self.P is None:
            raise RuntimeError("Need stationary distribution and stochastic matrix.")
        rev_P = time_reversed_transition_matrix(self.P, self.pi)
        self.Pr = 0.5 * (self.P + rev_P)
        return self.Pr

    def implied_timescales(self, tau: float) -> NDArray[np.float64]:
        evals = np.linalg.eigvals(self.P)
        evals = np.real(evals)
        evals = evals[np.argsort(-evals)]
        return -tau / np.log(np.clip(evals[1:], 1e-15, 1 - 1e-15))

    def compute_spectrum(self) -> None:
        # right: P r = λ r
        vals_r, vecs_r = np.linalg.eig(self.P)
        # left:  l^T P = λ l^T  <=>  P^T l = λ l
        vals_l, vecs_l = np.linalg.eig(self.P.T)
        # sort both with the same rule
        idx_r = _sort_idx(vals_r)
        idx_l = _sort_idx(vals_l)

        self.eigvals = np.real(vals_r[idx_r])
        self.l_eigvals = np.real(vals_l[idx_r])

        self.right_eigvecs = np.real(vecs_r[:, idx_r]) # columns = right eigenvectors
        self.left_eigvecs  = np.real(vecs_l[:, idx_l]) # columns = left  eigenvectors
        # convenience: slow modes exclude λ≈1 (last after this sorting)
        self.slow_modes = self.right_eigvecs[:, :-1]
        self.slow_mode  = self.slow_modes[:, -1] if self.slow_modes.size else None

    def compute_tr_spectrum(self) -> None:
        if self.Pr is None:
            raise RuntimeError("Run reversibilized_matrix first.")

        vals_r, vecs_r = np.linalg.eig(self.Pr)
        vals_l, vecs_l = np.linalg.eig(self.Pr.T)

        idx_r = _sort_idx(vals_r)
        idx_l = _sort_idx(vals_l)

        self.tr_eigvals = np.real(vals_l[idx_r])
        self.tr_right_eigvecs = np.real(vecs_r[:, idx_r])
        self.tr_left_eigvecs  = np.real(vecs_l[:, idx_l])

        self.tr_slow_modes = self.tr_right_eigvecs[:, :-1]
        self.tr_slow_mode  = self.tr_slow_modes[:, -1] if self.tr_slow_modes.size else None


    def compute_metastability(self,time_reversed = True) -> None:
        if time_reversed:
            slow = getattr(self, "tr_slow_modes", None)
        else:
            slow = getattr(self, "slow_modes", None)
        if slow is None or slow.size == 0:
            raise RuntimeError("Compute spectrum first (no slow modes).")
        slow_mode = slow[:, -1]
        self.thresholds = np.linspace(slow_mode.min(), slow_mode.max(), 100)
        meta_in, meta_out = [], []
        for t in self.thresholds:
            A = np.where(slow_mode >= t)[0]
            B = np.where(slow_mode < t)[0]
            meta_in.append(metastability(self.P, self.pi, A))
            meta_out.append(metastability(self.P, self.pi, B))
        self.meta_in = np.array(meta_in)
        self.meta_out = np.array(meta_out)
    
    def compute_entropy_rate(self) -> float:
        return entropy_rate(self.P, self.pi)



class Markov(StochasticMatrix):
    def __init__(self, traj_labels: List[NDArray[np.int_]], n_clusters: int, tau: int = 1) -> None:
        self.labels = traj_labels
        self.n_clusters = n_clusters
        self.tau = tau
        self.state: Optional[int] = None

        self.make_transition_matrix()

    def make_transition_matrix(self) -> None:
        C = count_transitions(
            self.labels,
            self.n_clusters,
            tau=self.tau
        )
        self.C = C
        with np.errstate(divide="ignore", invalid="ignore"):
            P = C / C.sum(axis=1, keepdims=True)
        P[np.isnan(P)] = 0.0
        super().__init__(P)
    
    def initialize_state(self):
        self.state = np.random.randint(0, self.n_clusters)
    
    def make_transition(self) -> int:
        """ given a current state : a cluster id, returns the id of the next cluster, selected according to the transition matrix."""
        if self.state is None:
            raise RuntimeError("Need to initialize the state first.")
        if self.P is None:
            raise RuntimeError("Need to make the transition matrix first.")
        cum_prob_array = np.cumsum(self.P[self.state])
        rd = np.random.randint(0, 1000) / 1000.0
        self.state = np.searchsorted(cum_prob_array, rd, side="right")
        return self.state
    
    def build_trajectory(self,T_tot:int)->np.ndarray:
        res = list()
        # This K is not defined anymore, I will set it to 1, but it should be fixed.
        # N_mkv_steps = T_tot//self.K 
        N_mkv_steps = T_tot
        self.initialize_state()
        res.append(self.state)
        for step in range(N_mkv_steps):
            res.append(self.make_transition())
        return np.array(res)

    def _build_sub_model(self, subset_indices: np.ndarray) -> Optional[StochasticMatrix]:
        """
        Builds a new StochasticMatrix for a subset of states.
        Internal helper for recursive_partition.
        """
        if self.C is None:
            raise RuntimeError("Count matrix 'C' not found. Ensure 'make_transition_matrix' has been run.")
        if subset_indices.size <= 1:
            return None # Cannot split a single state or an empty set

        # Extract the sub-count-matrix
        C_sub = self.C[np.ix_(subset_indices, subset_indices)]
        
        row_sums = C_sub.sum(axis=1, keepdims=True)
        
        # Handle states that only transition *out* of the subset (row_sum=0)
        # These are "leaky" states. We'll force a self-loop (P_ii = 1)
        # to keep the sub-matrix stochastic.
        if (row_sums == 0).any():
            leaky_rows_idx = np.where(row_sums == 0)[0]
            
            # Set diagonal element to 1 for these rows in C_sub
            C_sub[leaky_rows_idx, leaky_rows_idx] = 1.0
            row_sums[leaky_rows_idx] = 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            P_sub = C_sub / row_sums
        P_sub[np.isnan(P_sub)] = 0.0
        
        model = StochasticMatrix(P_sub)
        
        # Check if stationary distribution exists (e.g., if sub-model is disconnected)
        if model.pi is None or np.isnan(model.pi).any() or model.pi.sum() < 0.99:
             print(f"Warning: Could not find valid stationary distribution for subset {subset_indices}.")
             return None
             
        return model
            
    def recursive_partition(self, n_states: int, use_tr: bool = True, reduced_system:Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Iteratively partitions the state space into N macro-states using
        spectral bisection based on metastability.

        Parameters
        ----------
        n_states : int
            The target number of macro-states.
        use_tr : bool, optional
            Whether to use the reversibilized matrix (True) or the
            original transition matrix (False) for finding the slow mode.
            Default is True.
        reduced_system: np.ndaray, optional
            If an array is passed, it uses only the index passed
            to perform a partition.

        Returns
        -------
        List[np.ndarray]
            A list of arrays. Each array contains the original cluster
            indices (0 to n_clusters-1) that form a macro-state.
        """
        if self.C is None:
            raise RuntimeError("Count matrix 'C' not found. Call 'make_transition_matrix' first.")
        
        n_clusters = self.C.shape[0]
        
        # This list will hold the *final* macro-states
        final_partitions: List[np.ndarray] = []
        
        # This list holds macro-states that are *pending* splitting.
        # Start with the full set of all original cluster indices.
        if reduced_system is None:
            pending_partitions: List[np.ndarray] = [np.arange(n_clusters)]
        else:
            pending_partitions: List[np.ndarray] = [reduced_system]
        while len(final_partitions) + len(pending_partitions) < n_states:
            if not pending_partitions:
                print(f"Stopping partition: No more splittable partitions. Found {len(final_partitions)} states.")
                break
                
            # Pick the largest partition to split next (a good heuristic)
            pending_partitions.sort(key=len, reverse=True)
            current_partition_indices = pending_partitions.pop(0) # Get largest
            
            if current_partition_indices.size <= 1:
                # This partition is a single state, cannot be split.
                final_partitions.append(current_partition_indices)
                continue

            # 1. Build the sub-model for this partition
            model = self._build_sub_model(current_partition_indices)
            
            if model is None:
                print(f"Could not build sub-model for partition. Treating as final.")
                final_partitions.append(current_partition_indices)
                continue
                
            # 2. Compute spectrum and slow mode for the sub-model
            try:
                if use_tr:
                    model.reversibilized_matrix()
                    model.compute_tr_spectrum()
                    slow_mode = model.tr_slow_mode
                else:
                    model.compute_spectrum()
                    slow_mode = model.slow_mode
            except (np.linalg.LinAlgError, RuntimeError) as e:
                print(f"Error computing spectrum for partition: {e}. Treating as final.")
                final_partitions.append(current_partition_indices)
                continue
                
            if slow_mode is None:
                print(f"No slow mode found for sub-model. Treating partition as final.")
                final_partitions.append(current_partition_indices)
                continue

            # 3. Find optimal split for the sub-model
            try:
                model.compute_metastability(time_reversed=use_tr)
            except RuntimeError as e:
                print(f"Error computing metastability: {e}. Treating partition as final.")
                final_partitions.append(current_partition_indices)
                continue
            
            if model.meta_in is None or model.meta_out is None or model.thresholds is None:
                 print(f"Metastability computation failed. Treating partition as final.")
                 final_partitions.append(current_partition_indices)
                 continue

            min_meta = [min(a, b) for a, b in zip(model.meta_in, model.meta_out)]
            best_idx = np.argmax(min_meta)
            
            # Check if split is meaningful (metastability > 0)
            if min_meta[best_idx] <= 1e-9: # Use a small epsilon
                print(f"No meaningful split found (max min-metastability <= 0). Treating partition as final.")
                print(min_meta)
                final_partitions.append(current_partition_indices)
                continue

            best_threshold = model.thresholds[best_idx]
            
            # Get split *relative to the sub-model* (indices from 0 to len(slow_mode)-1)
            A_sub_indices = np.where(slow_mode >= best_threshold)[0]
            B_sub_indices = np.where(slow_mode < best_threshold)[0]

            if A_sub_indices.size == 0 or B_sub_indices.size == 0:
                print(f"Split resulted in an empty set. Treating partition as final.")
                final_partitions.append(current_partition_indices)
                continue
                
            # 4. Map sub-model indices back to *original* cluster indices
            new_partition_A = current_partition_indices[A_sub_indices]
            new_partition_B = current_partition_indices[B_sub_indices]
            
            # 5. Add new partitions to the pending list to be split further
            pending_partitions.append(new_partition_A)
            pending_partitions.append(new_partition_B)
            
            # print(f"Split partition (size {current_partition_indices.size}) into "
            #       f"{new_partition_A.size} and {new_partition_B.size} states.")

        # When loop finishes, all remaining pending partitions are also final
        final_partitions.extend(pending_partitions)
        
        #print(f"Finished partitioning. Found {len(final_partitions)} macro-states.")
        return final_partitions

class MarkovFromEmbedding(Markov):
    def __init__(self, embedding: EmbeddingBase, tau: int = 1) -> None:
        if embedding.labels is None:
            raise RuntimeError("Embedding must have cluster labels.")
        
        labels = embedding.labels
        n_clusters = embedding.n_clusters

        # store the shape of the embedding list to compute the transition correctly:
        embedding_shape = np.zeros(len(embedding.embedding_matrix),dtype=int)
        for idx in range(len(embedding.embedding_matrix)):
            embedding_shape[idx] = embedding.embedding_matrix[idx].shape[0]
        
        reshaped_labels = []
        current_pos = 0
        for length in embedding_shape:
            reshaped_labels.append(labels[current_pos : current_pos + length])
            current_pos += length
        
        super().__init__(reshaped_labels, n_clusters, tau)
        self.K = embedding.K

class MultiMarkov:
    def __init__(self, cluster: Cluster, tau: int = 1) -> None:
        self.models = []
        for traj_group in cluster.labels:
            model = Markov(traj_group, cluster.n_clusters, tau)
            self.models.append(model)

    def get_stochastic_matrices(self) -> List[NDArray[np.float64]]:
        return [model.P for model in self.models]

    def __getitem__(self, index: int) -> Markov:
        return self.models[index]

    def __len__(self) -> int:
        return len(self.models)