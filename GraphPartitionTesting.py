from GraphPartition import GraphPartition, random_graph, random_clustered_graph, Partition
import matplotlib.pyplot as plt
import numpy as np
import sys
import networkx as nx
import pylab as pyplt

def plot_graph_partition(G : nx.Graph, partition : Partition, name):
    Gs = [G.subgraph(b) for b in partition.buckets()]
    posns = [nx.nx_agraph.graphviz_layout(g, prog='dot') for g in Gs]
    colors = 'red blue green orange yellow purple'.split(' ')
    color_map = ['red']*G.number_of_nodes()
    for i,b in enumerate(partition.buckets()):
        for u in b:
            color_map[u] = colors[i]

    # Shift graph2
    shiftx, shifty = 200,400
    for i,pos in enumerate(posns):
        sx,sy = shiftx*i, shifty*(i%2)
        for key in pos:
            pos[key] = (pos[key][0] + sx, pos[key][1] + sy)

    # Combine the graphs and remove all edges
    union_g : nx.Graph = Gs[0]
    for g in Gs[1:]:
        union_g = nx.disjoint_union(union_g, g)
        for node in g.nodes:
            for neighbor in G.neighbors(node):
                if union_g.has_node(neighbor) and not g.has_node(neighbor):
                    if union_g.has_node(node) and union_g.has_node(neighbor):
                        union_g.add_edge(node,neighbor,weight = G.edges[node,neighbor]['weight'])

    union_posns = nx.nx_agraph.graphviz_layout(union_g, prog='dot')
    print(union_g.number_of_nodes())
    for posn in posns:
        for key in posn:
            union_posns[key]=posn[key]

    edge_colors = [union_g.edges[e]['weight'] for e in union_g.edges()]
    nx.draw_networkx(union_g, pos=union_posns, node_color = color_map, edge_color=edge_colors)

    pyplt.axis('off')
    pyplt.show()
    pyplt.savefig(f'./{name}')

sys.setrecursionlimit(10000)
n, k, p = 30, 3, 0.4
trials, attempts = 1, 1
np.random.seed(1)
# G = random_graph(n=n, p=p)
G = random_clustered_graph(n=30, k=3)
avgs = []
iterations = 50 * (2 ** np.arange(trials))
for iters in iterations:
    obj_vals = []
    for i in range(attempts):
        algo = GraphPartition()
        finish, df = algo.MCMC(G, k, n_rounds=iters)
        obj_vals.append(algo.eval_partition(G, finish))
    avgs.append(np.mean(obj_vals))
    #plot partitions if interested
    #plot_graph_partition(G,finish,name='test')
    print(np.mean(obj_vals))

plt.clf()
plt.plot(iterations, avgs, 'b')
plt.xlabel('iterations')
plt.ylabel('avg objective value')
plt.title('Avg objective value vs iterations')
plt.savefig("MCMC.jpg")