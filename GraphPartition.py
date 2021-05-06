from typing import Set, List, Iterable

import networkx as nx
import numpy as np
from copy import deepcopy


def random_graph(n=20, p=0.5, min_wt=0, max_wt=1) -> nx.Graph:
    G = nx.Graph()
    # add nodes
    for i in range(n):
        G.add_node(i)

    # add edges
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.uniform() < p:
                G.add_edge(i, j, weight=np.random.uniform(min_wt, max_wt))
            else:
                G.add_edge(i, j, weight=0)
    return G


def random_clustered_graph(n=30, p=0.5, k=5, min_wt=0, max_wt=1) -> nx.Graph:
    """
    Generated a random-like graph with some (unknown) pre-specified clusters
    """
    chunk_size = n // k
    remainder = n - chunk_size * k

    G = nx.Graph()
    # add nodes
    for i in range(n):
        G.add_node(i)

    # add cluster edges:
    s = 0
    for _ in range(k):
        e = s + chunk_size + 1 if remainder > 0 else s + chunk_size
        remainder -= 1
        for i in range(s, e):
            for j in range(i + 1, e):
                G.add_edge(i, j, weight=np.random.uniform(max_wt / 2, max_wt))
        s = e

    # add edges outside clusters
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.uniform() < p and not G.has_edge(i, j):
                G.add_edge(i, j, weight=np.random.uniform(min_wt, max_wt))
            else:
                G.add_edge(i, j, weight=0)
    return G


class GraphPartition:

    def __init__(self):
        pass

    def get_starting_point(self, G: nx.Graph, k: int) -> 'Partition':
        """
        Should return a starting point for the MCMC
        simulation. i.e. you should return some partition
        of the vertices into k classes
        :param G:
        :param k:
        :return:
        """
        return Partition(list(G.nodes), k)

    def get_edge_crossing_weights(self, G: nx.Graph, partition: 'Partition'):
        """
        Get the sum of the edge weights crossing between all pairs of
        vertices in distinct partitions.

        :param G:
        :param partition:
        :return: the sum of the weights of edges crossing between distinct partitions
        """
        #TODO pass

    def get_internal_weights(self, G: nx.Graph, partition: 'Partition') -> Iterable:
        """
        get the weight of the edges contained entirely in the same
        bucket of the partition, for each bucket in the partition
        :param G:
        :param partition:
        :return: an array of length k, where A[i] is the sum of the
        edge weights between vertices in bucket i of the partition
        """
        #TODO pass

    def eval_partition(self, G: nx.Graph, partition: 'Partition', Z=None):
        """
        this function will evaluate a partition of n vertices with k buckets
        based on the weight of edges crossing between the buckets, and the
        similarity of bucket sizes.

        :param G:
        :param partition:
        :return:
        """
        crossing_weight = self.get_edge_crossing_weights(G, partition)
        internal_weights = self.get_internal_weights(G, partition)
        Z = Z if Z is not None else 1
        k = partition.k
        scale = partition.n / k
        return crossing_weight - Z * np.prod(np.array(internal_weights) / scale)

    def apply_move(self, partition: 'Partition', elts, buckets):
        """
        apply the move - rearrange the partition so that the two elements
        are now mapped to the corresponding buckets in the partition

        :param partition:
        :param elts:
        :param buckets:
        :return:
        """
        partition.rearrange(elts, buckets)

    def eval_move(self, G: nx.Graph, partition: 'Partition', elts, buckets):
        """
        should return the change in the objective function after moving each element
        in elts to the corresponding bucket in buckets.
        i.e. the difference in eval_partition(curr) and eval_partition(new)
        where new is the partition after applying the move.
        """
        # Fast implementation :

        # the contribution from moving an element to a different bucket
        # can be calculated by
        # (1) finding the weight of the edges from this element to the vertices
        # in it's current bucket
        # (2) Finding the weight of the edges from this element to vertices in the new bucket
        # (3) subtracting (1) from (3)

        # With these contributions accounted for, you may still have to adjust for
        # the initial/ final placement of the two elements.

        # Slow Implementation :
        # alternatively (this is fine, but will be slower), you can simply make the
        # the change, and compare the function before and after making this change.
        # be sure to change the partition back to the original state before exiting
        # though.
        #TODO
        pass

    def propose_move(self, G: nx.Graph, partition):
        """
        propose a move - choose two vertices in G and
        assign them to buckets (possibly the same bucket they are already in)
        You can choose to do this however you'd like, but there should be a
        randomized component
        This can be as simple as choosing two vertices at random and two buckets at random
        and proposing this as a move.
        :param G:
        :param partition:
        :return: a list containing the chosen vertices, and a list
        """
        #TODO
        pass

    def transitionQ(self, energy_change, K=1):
        r = np.random.uniform(0, 1)
        t = np.exp(energy_change / K) > r
        return t

    def MCMC(self, G: nx.Graph, k, start=None, n_rounds=1000):
        # get a starting point
        if start is None:
            start = self.get_starting_point(G, k)
        f0 = self.eval_partition(G, start)
        finish = self._MCMC(G, start, n_rounds=n_rounds)
        f1 = self.eval_partition(G, finish)
        return finish, f0 - f1

    def _MCMC(self, G: nx.Graph, partition: 'Partition',
              n_rounds=1000, iter=0) -> 'Partition':
        """
        If you want, you can make this a simulated annealing algorithm by
        reducing the value of k proportional to the number of remaining iterations.
        :param G:
        :param partition:
        :param n_rounds:
        :param K:
        :param iter:
        :return:
        """

        if iter > n_rounds:
            return partition
        proposed_verts, proposed_buckets = self.propose_move(G, partition)
        delta_E = self.eval_move(G, partition, proposed_verts, proposed_buckets)
        if self.transitionQ(delta_E, K=1/np.log(iter+3)):
            self.apply_move(partition, proposed_verts, proposed_buckets)

        return self._MCMC(G, partition, n_rounds=n_rounds, iter=iter + 1)


class Partition:

    def __init__(self, elts, k, partition: List[Set] = None):
        self.k = k
        self.n = len(elts)
        self.elts = elts
        self.partition = random_partition(elts, k)
        if partition is not None:
            self.partition = partition
        self.partition_inverse = partition_inverse(self.partition)

    def bucket_sizes(self):
        return [len(x) for x in self.partition]

    def buckets(self):
        return [list(x) for x in self.partition]

    def same_bucketQ(self, i, j):
        """
        returns true iff i and j belong are in the same bucket
        according to this partition
        """
        assert 0 <= i < self.n
        assert 0 <= j <= self.n
        bucket_i = self.partition_inverse[i]
        bucket_j = self.partition_inverse[j]
        assert 0 <= bucket_i < self.k
        assert 0 <= bucket_j < self.k
        return bucket_i == bucket_j

    def rearrange(self, elts, new_buckets):
        """
        reorders the partition by removing elements in elts from their
        current buckets, and adding them to the corresponding bucket in
        new_buckets
        Rearranges the inverse of the partition to reflect the changes as well

        :param elts:
        :param new_buckets:
        :return:
        """
        if not isinstance(elts, Iterable):
            elts = [elts]
        if not isinstance(new_buckets, Iterable):
            new_buckets = [new_buckets]

        assert len(new_buckets) == len(elts)

        old_buckets = [self.partition_inverse[u] for u in elts]
        data = zip(elts, old_buckets, new_buckets)
        for elt, old_bucket, new_bucket in data:
            assert elt in self.partition[old_bucket]
            self.partition[old_bucket].remove(elt)
            self.partition[new_bucket].add(elt)
            self.partition_inverse[elt] = new_bucket

    def get_bucket(self, b):
        assert 0 <= b < self.k
        return self.partition[b]

    def get_bucket_idx(self, elt):
        assert elt in self.partition_inverse
        return self.partition_inverse[elt]

    def __len__(self):
        return len(self.partition)

    def clone(self):
        partition = list(self.partition)
        return Partition(self.elts, self.k, partition=deepcopy(partition))

    def __str__(self):
        return self.partition.__str__()


# randomly group elements into k partitions
def random_partition(elts, k) -> List[Set]:
    n = len(elts)
    rng = np.arange(n)
    temp = [x for x in elts]
    perm = np.random.permutation(rng)
    partition = [set() for _ in range(k)]

    for i, p in enumerate(perm):
        partition[i % k].add(temp[p])

    return partition


# return a map from each element to its partition index
def partition_inverse(partition: List[Set]):
    inverse = {}
    for i, bucket in enumerate(partition):
        for elt in bucket:
            inverse[elt] = i
    for i in inverse:
        assert 0 <= inverse[i] < len(partition)
    return inverse
