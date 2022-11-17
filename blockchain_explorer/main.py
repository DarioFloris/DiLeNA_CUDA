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


import sys, requests, os
from dotenv import load_dotenv
from tqdm import tqdm
from functools import partial
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import multiprocessing as mp
import blockchain_explorer as be
import pandas as pd


__version__ = '1.0.0'
weighted_graph = False
filename = None

crypto_list = {
    "btc"       : ['bitcoin','btc'], 
    "zcash"     : ['zcash', 'zec'],
    "ltc"       : ['litecoin', 'ltc'],
    "doge"      : ['dogecoin', 'doge'],
    "dash"      : ['dash', 'dash'],
    "eth"       : ['ethereum', None],
    "xrp"       : ['ripple', None]
}

def configure(dir_name, sub_dir, path='../graphs/'):
    load_dotenv()
    if not os.path.exists(path):
        os.mkdir(path)
    
    if not os.path.exists(path+dir_name):
        os.mkdir(path+dir_name)
    
    if not os.path.exists(path+dir_name+'/'+sub_dir):
        os.mkdir(path+dir_name+'/'+sub_dir)


def check_dlt(crypto_name):
    try:
        return crypto_list[crypto_name] 
    except KeyError:
        sys.exit('DLT not recognized')


def init_processes(target_dlt, weight, indexes, target_dlt_sc=None, processes=1):
    session = prepare_sessions()

    if target_dlt_sc is not None:
        get_tx_partial = partial(be.tx_from_sochain, dlt=target_dlt, session=session, weight=weight)
    elif target_dlt in 'ethereum':
        get_tx_partial = partial(be.tx_from_etherscan, session=session, weight=weight)
    
    pbar = tqdm(total=len(indexes), postfix='blocks', bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} {elapsed}', dynamic_ncols=True, ncols=50)
    result = []
    with mp.Pool(processes=processes) as pool:
        for batch in pool.imap_unordered(get_tx_partial, indexes):
            result.append(batch)
            pbar.update()

    pool.close()
    pool.join()
    pbar.close()

    return result


def write_file(dataset, path_to_file='../graphs/'):
    network_set = set()
    txs = []
    network_dict = {}
    for batch in dataset:
        for node in batch[0]:
            network_set.add(node)
        for tx in batch[1]:
            txs.append(tx)
    print('*Vertices: ', str(len(network_set)), ' *Arcs: ', str(len(txs)))    

    if len(network_set) > 0:
        with open(path_to_file +'_log.txt', 'w') as f:
            print('*Vertices:', str(len(network_set)), '*Archs:', str(len(txs)), sep= ' ', file=f)
        f.close()
        
        with open(path_to_file +'vertices.csv', 'w') as f:
            print('id', 'address', sep=',', file=f)
            for item in enumerate(network_set):
                network_dict[item[1]] = item[0]
                print(str(item[0]), str(item[1]), sep=',', file=f)
        f.close()
        with open(path_to_file+'network.csv', 'w') as f:
            print('sender', 'receiver', 'amount\n', sep=',', file=f)
            for t in txs:
                print(network_dict[t.sender], network_dict[t.receiver], t.amount, sep=',', file=f)
        f.close()
   


def prepare_sessions():
    s = requests.session()
    retry = Retry(connect= 5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)

    return s

usage_msg = """\

Main usage = > blockchain_explorer.py -dlt name -start "YYYY-MM-DD" -end "YYYY-MM-DD" -crawl [options]

options:
    -h      --help
    -v      --version
    -weight             transaction's value will be used as weight in the graph
    -cores              cores number for multiprocessing purposes

Legend
    -dlt                    bitcoin, etc..
    -start                  datetime as YYYY-MM-DD
    -end                    datetime as YYYY-MM-DD

"""
if __name__ == "__main__":
    if len(sys.argv) < 8 or '-h' in sys.argv or '--help' in sys.argv:
        sys.exit(usage_msg)
    if '-v' in sys.argv or '--version' in sys.argv:
        sys.exit('Blockchain_explorer.py ' + __version__)
    if '-dlt' in sys.argv:
        index = sys.argv.index('-dlt')
        dlt_name = sys.argv[index+1]
    if '-crawl' in sys.argv:
        crawler = True
    if '-start' in sys.argv:
        index = sys.argv.index('-start')
        start = sys.argv[index+1]
    if '-end' in sys.argv:
        index = sys.argv.index('-end')
        end = sys.argv[index+1]
    if '-weight' in sys.argv:
        weighted_graph = True
    if '-cores' in sys.argv:
        index = sys.argv.index('-cores')
        cores = int(sys.argv[index+1])
    

    child_dir = start +'_'+ end +'/'
    dlt_bc, dlt_sc_tx = check_dlt(dlt_name)
    configure(dlt_bc, child_dir)


    if dlt_name in 'xrp' or crawler:    
        print('Looking for block\'s indexes...', end='')
        first_block, last_block = be.crawl(start, end, dlt_bc)
    elif dlt_name not in 'xrp' and not crawler:
        print('Looking for', dlt_bc, 'blocks in the dump...', end='')
        first_block, last_block = be.gz_dump(start, end, dlt_bc)
    else:
        sys.exit('I don\'t know how to accomplish this task.')
    print('Done')    


    ids = [i for i in range(first_block, last_block+1)]
    print('Starting the download of the transactions...')
    data = init_processes(target_dlt=dlt_bc, weight=weighted_graph, target_dlt_sc=dlt_sc_tx, indexes=ids, processes=cores)
    print('Done')
    print('Writing to file...', end='')
    path_to_file = '../graphs/' + dlt_bc +'/'+ child_dir
    write_file(data, path_to_file=path_to_file)
    print('Done')
    