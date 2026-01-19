import numpy as np
import awkward as ak
from sklearn.cluster import KMeans
import umap

class Cluster:
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        self.embs = list()

    def add_embedding(self,embedding):
        self.embs.append(embedding)
    def add_embeddings(self,embeddings):
        self.embs += embeddings

    def set_shape_and_flatten(self):
        # concatenate and flatten all the embedding matrices
        self.flat_data = np.concatenate([emb.flatten_embedding_matrix for emb in self.embs])
        
        # Save the "Map": 
        # How many points in each flatten embedding matrix? (e.g., [10, 15, 10...])
        self.track_lengths = [len(t) for emb in self.embs for t in emb.embedding_matrix]
        
        # How many tracks in each group? (e.g., [5, 2, 8...])
        self.group_lengths = [len(emb.embedding_matrix) for emb in self.embs]
        
    #def reshape_flat_array(self,flat_array: np.ndarray):
        # Step A: Slice the flat array into individual tracks
        # We use cumsum to find the cut points
        # quantities -> [10, 20] -> cumsum -> [10, 30] -> split at indices
        #cut_indices_tracks = np.cumsum(self.track_lengths)[:-1]
        #list_of_tracks = np.split(flat_array, cut_indices_tracks)
        #
        ## Step B: Slice the list of tracks back into groups
        ## Note: We must treat list_of_tracks as an object array to split it effectively
        #cut_indices_groups = np.cumsum(self.group_lengths)[:-1]
        #
        ## This creates a list of arrays (groups), where each array contains tracks
        ## To get exactly "list of list of arrays", we map it back to lists
        #reshaped_array = np.split(np.array(list_of_tracks), cut_indices_groups)
        #return reshaped_array
    def reshape_flat_array(self, flat_array: np.ndarray):
        # Step A: Turn the flat array into a jagged array of tracks
        # ak.unflatten takes a flat array and a set of lengths
        tracks = ak.unflatten(flat_array, self.track_lengths)
        
        # Step B: Group those tracks into your higher-level groups
        # We unflatten the 'tracks' array using the group_lengths
        reshaped_array = ak.unflatten(tracks, self.group_lengths)
        
        return reshaped_array
    def clusterize(self,random_state=42,n_init='auto',max_iter=300,sample_weight=None):
        self.km = KMeans(n_clusters=int(self.n_clusters), random_state=random_state, n_init=n_init, max_iter=max_iter)
        self.km.fit(self.flat_data, sample_weight=sample_weight)

        self.flatten_labels = self.km.labels_.astype(int)
        self.cluster_centers_ = self.km.cluster_centers_

        #self.reshape_labels()
        self.labels = self.reshape_flat_array(self.flatten_labels)

    def fit_umap(self,n_neighbors: int = 15, min_dist: float = 0.1, with_cluster_centers=False)-> np.ndarray:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="euclidean")
        data_to_embed = self.flat_data
        if with_cluster_centers:
            data_to_embed = np.append(data_to_embed,self.cluster_centers_,axis=0)
        
        flat_reduced_pts = reducer.fit_transform(data_to_embed)
        
        if with_cluster_centers:
            self.reduced_centers = flat_reduced_pts[-self.cluster_centers_.shape[0]:]
            flat_reduced_pts = flat_reduced_pts[:-self.cluster_centers_.shape[0]]
        self.reduced_pts = self.reshape_flat_array(flat_reduced_pts)

#    def compute_proportions(self):
#        """
#        Compute the proportion of each embedding instance in each cluster.
#
#        Returns
#        -------
#        proportions : numpy.ndarray
#            A 2D array of shape (n_clusters, n_embeddings) where
#            proportions[i, j] is the proportion of points from embedding j
#            in cluster i.
#        """
#        if not hasattr(self, 'flatten_labels'):
#            raise ValueError("Clustering has not been performed yet. Call clusterize() first.")
#
#        n_embeddings = len(self.embs)
#        n_clusters = self.n_clusters
#        counts = np.zeros((n_clusters, n_embeddings))
#
#        for i in range(n_embeddings):
#            # self.labels[i] is an array of arrays of labels for embedding i
#            # Concatenate them to get a flat array of labels for the whole embedding
#            labels_for_embedding = np.concatenate(self.labels[i])
#            
#            bincounts = np.bincount(labels_for_embedding.astype(int), minlength=n_clusters)
#            counts[:, i] = bincounts
#
#        cluster_totals = np.sum(counts, axis=1, keepdims=True)
#
#        # handle case where a cluster is empty to avoid division by zero
#        proportions = np.divide(counts, cluster_totals, out=np.zeros_like(counts, dtype=float), where=cluster_totals!=0)
#
#        return proportions
    def compute_proportions(self):
        if not hasattr(self, 'labels'): # Changed from flatten_labels to labels
            raise ValueError("Clustering has not been performed yet.")

        n_embeddings = len(self.embs)
        n_clusters = self.n_clusters
        counts = np.zeros((n_clusters, n_embeddings))

        for i in range(n_embeddings):
            # self.labels[i] is the nested structure for embedding i.
            # axis=None collapses all tracks into one long 1D array of labels.
            labels_for_embedding = ak.to_numpy(ak.flatten(self.labels[i], axis=None))
            
            # Now bincount works perfectly on the flat NumPy array
            bincounts = np.bincount(labels_for_embedding.astype(int), minlength=n_clusters)
            counts[:, i] = bincounts

        cluster_totals = np.sum(counts, axis=1, keepdims=True)
        proportions = np.divide(counts, cluster_totals, 
                                out=np.zeros_like(counts, dtype=float), 
                                where=cluster_totals!=0)

        return proportions
