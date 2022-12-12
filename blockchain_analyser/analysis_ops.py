from numba.cuda.random import create_xoroshiro128p_states
from cugraph import Graph, connected_components
import cuda_ops as CUDA
from numba import cuda
from math import ceil
import numpy as np
import cupy as cp
import logger
import cudf
import sys


GPU_MEM_LIMIT = (18*1024**3) / 1e9

def load_data(filepath) -> cudf.DataFrame:
    df_edges = cudf.read_csv(filepath, delimiter=',', names=['src','dst','wt'],
                             dtype=['int32','int32','float64'])                           
    df_edges.drop_duplicates(subset=['src', 'dst'], inplace=True)
    df_edges.dropna(axis=0, how='any', inplace=True)

    return df_edges



def build_graph(
    graph, 
    edgelist, 
    source='src',
    destination='dst',
    edge_attr='wt',
    renumber=True,
    store_transposed=False) -> Graph:
    """
    Wraps cugraph.Graph.from_cudf_edgelist to initialize a graph from edge 
    list. source argument is source column name and destination is destination 
    column name.
    renumbering maps source and destination vertices into an index of range 
    [0, V).
    edge_attr argument is the weights column name.

    Parameters
    ----------
    graph : cugraph.Graph
        An empty or yet initialized graph to be rebuilt with the given data
    
    edgelist : cudf.DataFrame
        A Dataframe containing informations about edges.
    
    soure : str or array-like, optional (default='src')
        source column name or array of column names

    destination : str or array-like, optional (default='dst')
        destination column name or array of column names
    
    edge_attr : str or None, optional (default='wt')
        weights column name
    
    renumber : bool, optional (default=True)
        Whether or not to renumber the source and the destination vertex IDs

    store_transposed : bool, option (default=False)
        Whether or not to store the transpose of the adjacency matrix
    
    Returns
    -------
    graph : cugraph.Graph
        cugraph Graph instance with connectivity information as an edge list
    """
    
    graph.from_cudf_edgelist(edgelist, source, destination, edge_attr, renumber, 
                             store_transposed)
    logger.log(f'Building graph completed')

    return graph



def view_edgelist(graph) -> cudf.DataFrame:
    """
    Display the edge list
    """
    res = graph.view_edge_list()
    return res



def nodes(graph) -> cp.array:
    """
    Returns all the nodes in the graph as a cupy.array
    """
    res = graph.nodes().sort_values(ascending=True).to_cupy()
    return res



def number_of_vertices(graph) -> int:
    """
    Return the number of nodes in the graph
    """
    res = graph.number_of_vertices()
    logger.log(f'Number of nodes calculated')
    return res



def number_of_edges(graph) -> int:
    """
    Returns the number of edges in the graph
    """
    res = graph.number_of_edges()
    logger.log(f'Number of edges calculated')
    return res



def degree(graph, mode='tot') -> cudf.DataFrame:
    """
    Compute the degree of all the nodes in the graph

    Parameters
    ----------
    graph : cugraph.Graph
        The graph containing edge information
    mode : str, 'tot' 'in' or 'out' default='tot'
        Specify what kind of degree to compute on each vertex
    
    Returns
    -------
    df : cudf.DataFrame
        GPU DataFrame containing the degree information
        df[vertex] : cudf.Series
        df[degree] : cudf.Series
    """
    if mode in 'tot': df = graph.degree()
    elif mode in 'in': df = graph.in_degree()
    elif mode in 'out': df = graph.out_degree()

    df = df.sort_values(by='vertex', ignore_index=True)
    logger.log(f'"{mode}" degree calculated')
    
    return df



def degree_distribution(n, df, mode='tot') -> cudf.DataFrame:
    """
    Compute the distribution of the degrees as percentage and
    display it as GPU DataFrame.

    Parameters
    ----------
    n : int
        The number of vertices
    df : cudf.DataFrame
        DataFrame containing the degree information of the vertices
    mode : 'tot', 'in' or 'out'
        Unnecessary parameter 

    Returns
    -------
    df_distribution : cudf.DataFrame
        GPU DataFrame with degree's distribution information
        df_distribution[degree] : cudf.Series
            Degree value
        df_distribution[count] : cudf.Series
            Number of nodes with the same degree
        df_distribution[percentage] : cudf.Series
            Percentage of nodes with the same degree
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
    """
    Filters the vertices of the given graph separating them in groups 
    identified through labels. The most represented group is the main 
    component.

    Parameters
    ----------
    graph : cugraph.Graph
        The graph object containing edge information
    edges : cudf.DataFrame
        The edge list of the given graph
    Returns
    -------
    edges_list : cudf.DataFrame
        cudf.DataFrame with edge list representing the graph's main component
    """
    df_components = connected_components(graph, connection='weak')
    target_label = df_components['labels'].mode()[0]
    df_nodes = df_components[df_components['labels'] == target_label]
    edges_list = edges.loc[edges['src'].isin(df_nodes['vertex'])]
    
    logger.log(f'Main component\'s edges calculated')
    return edges_list




def _compute_bounds(x, y, bytes_) -> int:
    """
    PAY ATTENTION
    -------------
    This is an internal experimental function, it allows to bypass the need of 
    a memory manager since CUDA technology has some memory restiction about 
    the actual amount of memory aviable with WSL (Windows Subsistem for Linux)
    Improvements of such features must be considered.

    Approximates the demand of memory for an object of size (x, y) in which 
    each element needs bytes_ to be stored.
    
    Parameters
    ----------
    x : int
        number of rows of the object
    y : int
        number of columns of the object
    bytes_ : int
        Number of bytes consumed by a single element to be stored/represented
        in memory
    
    Returns
    -------
    x : float
        Maximum number of rows allowed without hitting/exceed memory limits 
    """
    size = ((x * y * bytes_) / 1e9)
    if size > GPU_MEM_LIMIT:
        x = _compute_bounds(int(x/2), y, bytes_)
    return x




def _init_cc(n, batch_size, iteration, mod) -> tuple[int, int, int]:
    """
    Avg_clustering_coefficient's helper function.
    Computes ranges with which performe the core computation

    Parameters
    ----------
    n : int
        Represents upper limit of all the bounds
    batch_size : int
        Dimension of the batch
    iteration : int
        Multiplier to obtain the values of bottom and upper bounds correctly
    mod : int
        Represents the upper bound of the last batch if n cannot be evenly 
        divided.

    Returns
    -------
    start : int
        Bottom bound or first valid value to be used
    stop : int
        Upper bound or last valid value to be used
    M : int
        Batch size dimension used by specific objects
    """

    if (batch_size*iteration) <= n:
        start = 0 + (batch_size * (iteration-1))
        stop = start+batch_size
        M = batch_size
    else:
        start = 0 + (batch_size * (iteration-1))
        stop = start+mod
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
    """
    Computes the average clustering coefficient of a graph with given n 
    vertices. Sets up every object needed to perform the computation, then 
    uses CUDA Kernels with these objects as parameters.

    Parameters
    ----------
    n : int
        Number of vertices upon which to compute the average clustering 
        coefficient.
    src : cudf.Series
        GPU Series containing source's edge information
    dst : cudf.Series
        GPU Series containing destination's edge information
    df_degree : cudf.Series
        GPU Series containing degrees information
    N : int
        Maximum width of the adjacency lists.
    nodes_cp : cp.array
        GPU array containing vertices identifiers
        Mainly used when computing the average clustering coefficient of 
        Main components
    undirected : bool, optional (default=False)
        'True' value is not supported yet.
        Allow to compute the average clustering coefficient on both directed 
        and undirected graphs

    Returns
    -------
    result : float
        The average clustering coefficient of the entire graph with the given 
        vertices

    FIX ME
    Test the algorithm on graph which vertex IDs has been renumbered.
    After renumbering, get the edge list again, the degree dataframe and
    test the function without nodes_cp parameter.
    """
    if undirected:
        sys.exit(f'Unidrected parameter is not supported')

    local_ccs = cp.zeros((1,), dtype='float32')
    M = _compute_bounds(n, N, cp.dtype(cp.int32).itemsize)
    epochs = ceil(n / M)
    leftovers = n % M
     
    for i in range(1, epochs+1):  
        start, stop, M = _init_cc(n, M, i, leftovers)
        nodes = cp.arange(start, stop, 1)
        if nodes_cp is not None: nodes = nodes_cp[start : stop]
        matrix = cp.ones((M, N), dtype='int32')
        matrix = cp.negative(matrix)
        edgespernode = cp.zeros(M, dtype='int32')
#        reciprocal = cp.zeros(M, dtype='int32')
        start_ev = cuda.event()
        stop_ev = cuda.event()

        threadsperblock = 1024
        blockspergrid = (M + (threadsperblock - 1)) // threadsperblock
        threadsperblock_2D = (32, 32)
        blockspergrid_x = (N + (threadsperblock_2D[1] - 1)) // threadsperblock_2D[1]
        blockspergrid_y = (M + (threadsperblock_2D[0] - 1)) // threadsperblock_2D[0]
        blockspergrid_2D = (blockspergrid_x, blockspergrid_y)



        start_ev.record()
        CUDA.adj_list[blockspergrid, threadsperblock](nodes, src, dst, undirected, matrix)
#        CUDA.reciprocal_count[blockspergrid_2D, threadsperblock_2D](matrix, nodes, src, dst, M, N, reciprocal)
        CUDA.find_uv_edges[blockspergrid_2D, threadsperblock_2D](matrix, src, dst, M, N, edgespernode)       
        CUDA.lcc[blockspergrid, threadsperblock](nodes, edgespernode, df_degree, undirected, local_ccs)
        stop_ev.record()
        cuda.synchronize()

    elapsed_t = (cuda.event_elapsed_time(start_ev, stop_ev) / 1000) / 60
    result = local_ccs[0].get() / n
    logger.log(f'Average clustering coefficient calculcated. {result}')
    logger.log('Elapsed time: %.6f minutes' % elapsed_t)
    return result




def random_graph_generator(n, edges) -> cudf.DataFrame:
    """
    PAY ATTENTION
    -------------
    This function is experimental, deeper tests must be performed to ensure 
    stability, good result and good performances.

    Generates an edge list with ER model in order to build a random graph.

    Parameters
    ----------
    n : int
        Number of vertices of the original graph
    edges : int
        Number of edges of the original graph
    
    Return
    ------
    df : cudf.DataFrame
        GPU DataFrame containing edge information
    """
    logger.log(f'Generating random graph with ER model...')
    L = Graph(directed=True)
    df = cudf.DataFrame({'src': None, 'dst': None})
    p = edges / (n * (n-1))
    N = n
    M = _compute_bounds(n, N, cp.dtype(cp.int32).itemsize)
    epochs = ceil(n / M)
    leftovers = n % M
    epoch = 1

    while epoch <= epochs:
        L.clear()
        start, stop, M = _init_cc(n, M, epoch, leftovers)
        matrix = cp.zeros((M, N), dtype='int32')
        
        threadsperblock = 1024
        blockspergrid = (matrix.shape[0] + (threadsperblock - 1)) // threadsperblock
        rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid, seed=42)
        CUDA.gnp_erdos_renyi[blockspergrid, threadsperblock](p, rng_states, M, N, matrix)
        
        a_matrix = np.empty((M,N), dtype='int32')
        cp.asnumpy(matrix, stream=None, out=a_matrix)
        L.from_numpy_array(a_matrix)
        df_l = L.view_edge_list()
        df_l.pop('weights')
        size = len(df['src'])
        CUDA.align.forall(size)(df_l['src'], start)
        df = cudf.concat([df, df_l], ignore_index=True)

        del a_matrix
        del matrix
        epoch += 1
    
    df.dropna(inplace=True)
    logger.log(f'Random graph\'s edges list calculated')
    
    return df
