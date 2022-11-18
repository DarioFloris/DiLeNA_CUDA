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
__version__ = '0.4.0'

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
    dlt = check_dlt(dlt_name)
    path_to_file = check_fname(dlt, file_name)

#    logger.logger_config('')
#   Original network
    analysis_ = Analysis(direction=True)
    G = analysis_.get_graph()
    edges = ops.load_data(path_to_file)
    ops.build_graph(G, edges)

#    Collect graph's properties
    vertex_number = ops.number_of_vertices(G)
    edges_number = ops.number_of_edges(G)

    df_total_deg = ops.degree(G, 'tot')
    df_in_deg = ops.degree(G, 'in')
    df_out_deg = ops.degree(G, 'out')

    df_tot = ops.degree_distribution(vertex_number, df_total_deg, mode='tot')
    df_in = ops.degree_distribution(vertex_number, df_in_deg, mode='in')
    df_out = ops.degree_distribution(vertex_number, df_out_deg, mode='out')

#    compute_shortest_path_length(G, G.nodes().to_cupy())

    N = df_total_deg['degree'].max()
    avg_cc = ops.avg_clustering_coefficient(
        vertex_number, 
        edges['src'], 
        edges['dst'], 
        df_total_deg['degree'], 
        N, 
        undirected=False
    )

#   MAIN COMPONENT
    edges_mc = ops.build_main_weakly_connected_component_edges(G, edges)
    analysis_mc = Analysis(direction=True)
    G_mc = analysis_mc.get_graph()
    ops.build_graph(G_mc, edges_mc, renumber=True)

#    Collect main component's properties
    vertex_number_mc = ops.number_of_vertices(G_mc)
    edges_number_mc = ops.number_of_edges(G_mc)

    df_total_deg_mc = ops.degree(G_mc, 'tot')
    df_in_deg_mc = ops.degree(G_mc, 'in')
    df_out_deg_mc = ops.degree(G_mc, 'out')

    vertices_mc = ops.nodes(G_mc)
    
    df_tot_mc = ops.degree_distribution(vertex_number_mc, df_total_deg_mc, mode='tot')
    df_in_mc = ops.degree_distribution(vertex_number_mc, df_in_deg_mc, mode='in')
    df_out_mc = ops.degree_distribution(vertex_number_mc, df_out_deg_mc, mode='out')


    N = df_total_deg_mc['degree'].max()
    avg_cc_mc = ops.avg_clustering_coefficient(
        vertex_number_mc, 
        edges_mc['src'], 
        edges_mc['dst'], 
        df_total_deg_mc['degree'], 
        N,
        nodes_cp=vertices_mc, 
        undirected=False
    )


#   RANDOM GRAPH
    edges_rnd = ops.random_graph_generator(vertex_number, edges_number)
    analysis_rnd = Analysis(direction=True)
    G_rnd = analysis_rnd.get_graph()
    ops.build_graph(G_rnd, edges_rnd, edge_attr=None, renumber=False)
    
#   Collect Random Graph's properties
    vertex_number_rnd = ops.number_of_vertices(G_rnd)
    edges_number_rnd = ops.number_of_edges(G_rnd)

    df_tot_deg_rnd = ops.degree(G_rnd, 'tot')
    df_in_deg_rnd = ops.degree(G_rnd, 'in')
    df_out_deg_rnd = ops.degree(G_rnd, 'out')

    df_tot_rnd = ops.degree_distribution(vertex_number_rnd, df_tot_deg_rnd, mode='tot')
    df_in_rnd = ops.degree_distribution(vertex_number_rnd, df_in_deg_rnd, mode='in')
    df_out_rnd = ops.degree_distribution(vertex_number_rnd, df_out_deg_rnd, mode='out')

    N = df_tot_deg_rnd['degree'].max()
    avg_cc_rnd = ops.avg_clustering_coefficient(
        vertex_number_rnd, 
        edges_rnd['src'], 
        edges_rnd['dst'], 
        df_tot_deg_rnd['degree'], 
        N, 
        undirected=False
    ) 

#   RANDOM GRAPH's MAIN COMPONENT
    edges_rnd_mc = ops.build_main_weakly_connected_component_edges(G_rnd, edges_rnd)
    analysis_rnd_mc = Analysis(direction=True)
    G_rnd_mc = analysis_rnd_mc.get_graph()
    ops.build_graph(G_rnd_mc, edges_rnd_mc, edge_attr=None, renumber=False)

    vertex_number_rnd_mc = ops.number_of_vertices(G_rnd_mc)
    edges_number_rnd_mc = ops.number_of_edges(G_rnd_mc)

    df_total_deg_rnd_mc = ops.degree(G_rnd_mc, 'tot')
    df_in_deg_rnd_mc = ops.degree(G_rnd_mc, 'in')
    df_out_deg_rnd_mc = ops.degree(G_rnd_mc, 'out')

    vertices_rnd_mc = ops.nodes(G_rnd_mc)

    df_tot_rnd_mc = ops.degree_distribution(vertex_number_rnd_mc, df_total_deg_rnd_mc, mode='tot')
    df_in_rnd_mc = ops.degree_distribution(vertex_number_rnd_mc, df_in_deg_rnd_mc, mode='in')
    df_out_rnd_mc = ops.degree_distribution(vertex_number_rnd_mc, df_out_deg_rnd_mc, mode='out')

    N = df_total_deg_rnd_mc['degree'].max()
    avg_cc_rnd_mc = ops.avg_clustering_coefficient(
        vertex_number_rnd_mc, 
        edges_rnd_mc['src'], 
        edges_rnd_mc['dst'],
        df_in_deg_rnd_mc['degree'],
        N,
        nodes_cp=vertices_rnd_mc,
        undirected=False
    )

    #TODO logger file creation
    #TODO json file creation containing analysis results