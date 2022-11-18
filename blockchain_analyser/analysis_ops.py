import cuda_ops as CUDA
from math import ceil
import cupy as cp
import cudf
from cugraph import Graph, connected_components
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np
import logger

GPU_MEM_LIMIT = (18*1024**3) / 1e9

def load_data(filepath):
        df_edges = cudf.read_csv(filepath, delimiter=',', names=['src','dst','wt'],
                                    dtype=['int32','int32','float64'])
                                    
        df_edges.drop_duplicates(subset=['src', 'dst'], inplace=True)
        df_edges.dropna(axis=0, how='any', inplace=True)

        return df_edges

def build_graph(
    graph, 
    edges, 
    source='src',
    destination='dst',
    edge_attr='wt',
    renumber=False,
    store_transposed=True
) -> Graph:

    graph.from_cudf_edgelist(edges, source, destination, edge_attr, renumber, 
                             store_transposed)

    logger.log(f'Building graph completed')
    return graph

def nodes(graph):
    vertices = graph.nodes().sort_values(ascending=True).to_cupy()
    return vertices

def number_of_vertices(graph):
    vertices = graph.number_of_vertices()
    logger.log(f'Number of nodes calculated')
    return vertices

def number_of_edges(graph):
    edges = graph.number_of_edges()
    logger.log(f'Number of edges calculated')
    return edges

def degree(graph, mode='tot'):

    if mode in 'tot': df = graph.degree()
    elif mode in 'in': df = graph.in_degree()
    elif mode in 'out': df = graph.out_degree()

    df = df.sort_values(by='vertex', ignore_index=True)
    logger.log(f'"{mode}" degree calculated')
    return df


def degree_distribution(n, df, mode='tot') -> cudf.DataFrame:
    """
    - df           cudf dataframe containing in/out/total degree per each node
    - n            number of vertices    
    - mode         tot OR in OR out degree to specify nothing(??????????????)      
    """
    
    degree_series = df['degree'].value_counts()
    df_distribution = cudf.DataFrame({'degree': degree_series.index.to_cupy(),
                                      'count': degree_series.to_cupy(),
                                      'percentage': 0.0})

    size = len(df_distribution)
    CUDA.compute_degree_distribution.forall(size)(n, df_distribution['count'],
                                                  df_distribution['percentage'])
    
    logger.log(f'"{mode}" degree distribution calculated')
    return df_distribution


def build_main_weakly_connected_component_edges(graph, edges) -> cudf.DataFrame:
    df_components = connected_components(graph, connection='weak')
    target_label = df_components['labels'].mode()[0]
    df_nodes = df_components[df_components['labels'] == target_label]
    edges_list = edges.loc[edges['src'].isin(df_nodes['vertex'])]
    
    logger.log(f'Main component\'s edges calculated')
    return edges_list




def compute_bounds(x, y, bytes_) -> int:
    size = ((x * y * bytes_) / 1e9)
    if size > GPU_MEM_LIMIT:
        x  = compute_bounds(int(x/2), y, bytes_)
    return x


def init_cc(n, batch_size, iteration, mod):

    """
    - n                 number of vertices of the graph
    - batch_size        range of nodes examined each epoch
    - iteration         current epoch
    - mod               the margin of n / batch_size
    """

    if (batch_size * iteration) <= n:
        start = 0 + (batch_size*(iteration - 1))
        stop = start + batch_size
        M = batch_size
    else:
        start = 0 + (batch_size*(iteration - 1))
        stop = start + mod
        M = mod
    print(start, stop)
    return start, stop, M
        
    
def avg_clustering_coefficient(
    n,
    src,
    dst,
    df_degree,
    N,
    nodes_cp=None,
    undirected=False

) -> float:


    local_ccs = cp.zeros((1,), dtype='float32')
    M = compute_bounds(n, N, cp.dtype(cp.int32).itemsize)
    epochs = ceil(n / M)
    leftovers = n % M
     
    for i in range(1, epochs+1):  
        start, stop, M = init_cc(n, M, i, leftovers)
        nodes = cp.arange(start, stop, 1)
        if nodes_cp is not None: nodes = nodes_cp[start : stop]
        matrix = cp.empty((M, N), dtype='int32')
        matrix.fill(-1)
        edgespernode = cp.zeros(M, dtype='int32')
        reciprocal = cp.zeros(M, dtype='int32')
        start_ev = cuda.event()
        stop_ev = cuda.event()

        threadsperblock = 1024
        blockspergrid = (M + (threadsperblock -1)) // threadsperblock
        threadsperblock_2D = (32, 32)
        blockspergrid_x = (N + (threadsperblock_2D[1] - 1)) // threadsperblock_2D[1]
        blockspergrid_y = (M + (threadsperblock_2D[0] - 1)) // threadsperblock_2D[0]
        blockspergrid_2D = (blockspergrid_x, blockspergrid_y)



        start_ev.record()
        CUDA.adj_list[blockspergrid, threadsperblock](nodes, src, dst, undirected, matrix)
        CUDA.reciprocal_count[blockspergrid_2D, threadsperblock_2D](matrix, nodes, src, dst, M, N, reciprocal)
        CUDA.find_uv_edges[blockspergrid_2D, threadsperblock_2D](matrix, src, dst, M, N, undirected, edgespernode)       
        CUDA.lcc[blockspergrid, threadsperblock](nodes, edgespernode, df_degree, reciprocal, undirected, local_ccs)
        stop_ev.record()
        cuda.synchronize()

    elapsed_t = (cuda.event_elapsed_time(start_ev, stop_ev) / 1000) / 60
    result = local_ccs[0].get() / n
    logger.log(f'Average clustering coefficient calculcated. {result}')
    logger.log('Elapsed time: %.6f minutes' % elapsed_t)
    return result


def random_graph_generator(n, edges) -> cudf.DataFrame:
    L = Graph(directed=True)
    df = cudf.DataFrame({'src': None, 'dst': None})
    p = edges / (n * (n - 1))
    N = n
    M = compute_bounds(n, N, cp.dtype(cp.int32).itemsize)
    epochs = ceil(n / M)
    leftovers = n % M
    epoch = 1

    while epoch <= epochs:
        L.clear()
        start, stop, M = init_cc(n, M, epoch, leftovers)
        matrix = cp.zeros((M, N), dtype='int32')
        threadsperblock = 1024
        blockspergrid = (matrix.shape[0] + (threadsperblock - 1)) // threadsperblock
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=42)
        CUDA.gnp_erdos_renyi[blockspergrid, threadsperblock](p, rng_states, M, N, matrix)
        a_matrix = np.empty((M,N), dtype='int32')
        cp.asnumpy(matrix, stream=None, out=a_matrix)
#        arrays.append(a_matrix)
        L.from_numpy_array(a_matrix)
        df_l = L.view_edge_list()
        df_l.pop('weights')
        size = len(df['src'])
        CUDA.align.forall(size)(df_l['src'], start)
        df = cudf.concat([df, df_l], ignore_index=True)
        del a_matrix
        del matrix
        epoch += 1
    
#    A = np.concatenate(arrays)
    df.dropna(inplace=True)

    logger.log(f'Random graph\'s edges list calculated')
    return df
