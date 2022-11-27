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

__version__ = '1.2.0'

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

def compute_graph(G, edgelist, is_main_comp=False):
    vertex_number_ = ops.number_of_vertices(G)
    edges_number_ = ops.number_of_edges(G)

    df_total_deg = ops.degree(G, 'tot')
    df_in_deg = ops.degree(G, 'in')
    df_out_deg = ops.degree(G, 'out')

    df_tot = ops.degree_distribution(vertex_number_, df_total_deg, mode='tot')
    df_in = ops.degree_distribution(vertex_number_, df_in_deg, mode='in')
    df_out = ops.degree_distribution(vertex_number_, df_out_deg, mode='out')
    vertices = ops.nodes(G) if is_main_comp else None
#    compute_shortest_path_length(G, G.nodes().to_cupy())

#   AVG_CC for directed graphs:
#   - N = total_deg['degree'] 
#   - df_out_deg['degree]

    N = df_out_deg['degree'].max()
    avg_cc_ = ops.avg_clustering_coefficient(
        vertex_number_, 
        edgelist['src'], 
        edgelist['dst'], 
        df_out_deg['degree'], 
        N,
        nodes_cp=vertices, 
        undirected=False
    )



def main(dltname, filename):
    dlt = check_dlt(dltname)
    path_to_file = check_fname(dlt, filename)
    logger.logger_config(filename)

    #   Original network
    analysis_ = Analysis(direction=True)
    G = analysis_.get_graph()
    edges_ = ops.load_data(path_to_file)
    ops.build_graph(G, edges_, renumber=False)
    compute_graph(G, edges_)
    
#   MAIN COMPONENT
    edges = ops.build_main_weakly_connected_component_edges(G, edges_)
    analysis_mc = Analysis(direction=True)
    G_mc = analysis_mc.get_graph()
    ops.build_graph(G_mc, edges, renumber=True)
    compute_graph(G_mc, edges, is_main_comp=True)

#   RANDOM GRAPH
    edges = ops.random_graph_generator(ops.number_of_vertices(G), ops.number_of_edges(G))
    analysis_rnd = Analysis(direction=True)
    G_rnd = analysis_rnd.get_graph()
    ops.build_graph(G_rnd, edges, edge_attr=None, renumber=False)  
    compute_graph(G_rnd, edges)

#   RANDOM GRAPH's MAIN COMPONENT
    edges = ops.build_main_weakly_connected_component_edges(G_rnd, edges)
    analysis_rnd_mc = Analysis(direction=True)
    G_rnd_mc = analysis_rnd_mc.get_graph()
    ops.build_graph(G_rnd_mc, edges, edge_attr=None, renumber=True)
    compute_graph(G_rnd_mc, edges, is_main_comp=True)

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