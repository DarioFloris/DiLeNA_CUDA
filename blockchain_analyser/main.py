''' 
##############################################################################
    DiLeNA_CUDA, Distributed Ledger Network Analyzer with CUDA technology
                        
                        http://pads.cs.unibo.it


    CUDA-based Software developed by
        Dario Floris                     <dario.floris2@studio.unibo.it


    Original Software developed by ANANSI research group:
        Gabriele D'angelo   <g.@unibo.it>
        Stefano Ferretti    <stefano.ferretti@unibo.it>
        Luca Serena         <luca.serena2@unibo.it>
        Mirko Zichichi      <mirko.zichichi@upm.es>

##############################################################################
'''

import sys, os
from analysis import Analysis
import analysis_ops as ops
import logger

__version__ = '1.0'

base_path = '../graphs/'

crypto_list = {
    "btc"       : 'bitcoin', 
    "zcash"     : 'zcash',
    "ltc"       : 'litecoin',
    "doge"      : 'dogecoin',
    "dash"      : 'dash',
    "eth"       : 'ethereum',
    "xrp"       : 'ripple'
}


def check_dlt(dlt) -> str:
    try:
        return crypto_list[dlt]
    except KeyError:
        sys.exit(f'DLT not recognized')

def check_fname(dlt, fname, base_path='graphs/'):
    full_path = base_path+dlt +'/'+ fname +'/network.csv'
    if not os.path.exists(full_path):
        sys.exit(f'{full_path} is not a valid path')
    else:
        return full_path



def main(dltname, filename):
    dlt = check_dlt(dltname)
    path_to_file = check_fname(dlt, filename)

    #   Original network
    analysis_ = Analysis(direction=True)
    G = analysis_.get_graph()
    edges_ = ops.load_data(path_to_file)
    ops.build_graph(G, edges_, renumber=True)

#    Collect graph's properties
    vertex_number_ = ops.number_of_vertices(G)
    edges_number_ = ops.number_of_edges(G)

    df_total_deg = ops.degree(G, 'tot')
    df_in_deg = ops.degree(G, 'in')
    df_out_deg = ops.degree(G, 'out')

    df_tot = ops.degree_distribution(vertex_number_, df_total_deg, mode='tot')
    df_in = ops.degree_distribution(vertex_number_, df_in_deg, mode='in')
    df_out = ops.degree_distribution(vertex_number_, df_out_deg, mode='out')

#    compute_shortest_path_length(G, G.nodes().to_cupy())

#   AVG_CC for directed graphs:
#   - N = total_deg['degree'] 
#   - df_out_deg['degree]

    N = df_out_deg['degree'].max()
    avg_cc_ = ops.avg_clustering_coefficient(
        vertex_number_, 
        edges_['src'], 
        edges_['dst'], 
        df_out_deg['degree'], 
        N, 
        undirected=False
    )

#   AVG_CC for undirected graphs:
    """
    FIX ME:

    Undirected graph needs the weight column list to be dropped before
    the next instructions take place

    G_undirected = G.to_undirected()
    edge_undi = ops.view_edgelist(G_undirected)

    f = ops.number_of_edges(G_undirected)
    v_undi = ops.number_of_vertices(G_undirected)
    e_undi = ops.number_of_edges(G_undirected)
    df_out_undi = ops.degree(G_undirected, 'out')
    N_undi = df_out_undi['degree'].max()
    avg_ = ops.avg_clustering_coefficient(
        v_undi,
        edge_undi['src'],
        edge_undi['dst'],
        df_out_undi['degree'],
        N_undi,
        undirected=True
    )
    """

#   MAIN COMPONENT
    edges = ops.build_main_weakly_connected_component_edges(G, edges_)
    analysis_mc = Analysis(direction=True)
    G = analysis_mc.get_graph()
    ops.build_graph(G, edges, renumber=True)

#    Collect main component's properties
    vertex_number = ops.number_of_vertices(G)
    edges_number = ops.number_of_edges(G)

    df_total_deg = ops.degree(G, 'tot')
    df_in_deg = ops.degree(G, 'in')
    df_out_deg = ops.degree(G, 'out')

    df_tot = ops.degree_distribution(vertex_number, df_total_deg, mode='tot')
    df_in = ops.degree_distribution(vertex_number, df_in_deg, mode='in')
    df_out = ops.degree_distribution(vertex_number, df_out_deg, mode='out')

    vertices = ops.nodes(G)

    N = df_out_deg['degree'].max()
    avg_cc_mc = ops.avg_clustering_coefficient(
        vertex_number, 
        edges['src'], 
        edges['dst'], 
        df_out_deg['degree'], 
        N,
        nodes_cp=vertices, 
        undirected=False
    )


#   RANDOM GRAPH
    edges = ops.random_graph_generator(vertex_number_, edges_number_)
    analysis_rnd = Analysis(direction=True)
    G = analysis_rnd.get_graph()
    ops.build_graph(G, edges, edge_attr=None, renumber=False)
    
#   Collect Random Graph's properties
    vertex_number = ops.number_of_vertices(G)
    edges_number = ops.number_of_edges(G)

    df_tot_deg = ops.degree(G, 'tot')
    df_in_deg = ops.degree(G, 'in')
    df_out_deg = ops.degree(G, 'out')

    df_tot = ops.degree_distribution(vertex_number, df_tot_deg, mode='tot')
    df_in = ops.degree_distribution(vertex_number, df_in_deg, mode='in')
    df_out = ops.degree_distribution(vertex_number, df_out_deg, mode='out')

    N = df_out_deg['degree'].max()
    avg_cc_rnd = ops.avg_clustering_coefficient(
        vertex_number, 
        edges['src'], 
        edges['dst'], 
        df_out_deg['degree'], 
        N, 
        undirected=False
    ) 

#   RANDOM GRAPH's MAIN COMPONENT
    edges = ops.build_main_weakly_connected_component_edges(G, edges)
    analysis_rnd_mc = Analysis(direction=True)
    G = analysis_rnd_mc.get_graph()
    ops.build_graph(G, edges, edge_attr=None, renumber=True)

    vertex_number = ops.number_of_vertices(G)
    edges_number = ops.number_of_edges(G)

    df_total_deg = ops.degree(G, 'tot')
    df_in_deg = ops.degree(G, 'in')
    df_out_deg = ops.degree(G, 'out')

    vertices = ops.nodes(G)
    df_tot = ops.degree_distribution(vertex_number, df_total_deg, mode='tot')
    df_in = ops.degree_distribution(vertex_number, df_in_deg, mode='in')
    df_out = ops.degree_distribution(vertex_number, df_out_deg, mode='out')

    N = df_out_deg['degree'].max()
    avg_cc_rnd_mc = ops.avg_clustering_coefficient(
        vertex_number, 
        edges['src'], 
        edges['dst'],
        df_out_deg['degree'],
        N,
        nodes_cp=vertices,
        undirected=False
    )
    
    print("Done")

usage_msg = """ \

Main usage => blockchain_analyser.py -s path_name -o path_name

Legend
    -s      source path to the data to be loaded
    -o      destination path to save analysis results
"""
if __name__ == '__main__':
#    if len(sys.argv) < 6 or '-h' in sys.argv or '--help' in sys.argv:
#        sys.exit(f'{usage_msg}')
#    if '-v' in sys.argv or '--version' in sys.argv:
#        sys.exit(f'Blockchain_analyser.py {__version__}')
#    if '-dlt' in sys.argv:
#        index = sys.argv.index('-dlt')
#        dlt_name = sys.argv[index+1]
#    if '-n' in sys.argv:
#        index = sys.argv.index('-n')
#        file_name = sys.argv[index+1]
    
    dlt_name = 'eth'
    file_name = '2020-01-01_2020-01-01'
    main(dlt_name, file_name)

    #TODO logger file creation
    #TODO json file creation containing analysis results