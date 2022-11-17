#from functools import partial
import gzip, io, requests, os, sys
from itertools import product
from time import sleep
import pandas as pd

API_RPS_LIMIT = 429
API_OK = 200

class Transaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

def crawl(str_start, str_end, dlt):   
    if dlt in 'ripple':
        pass
    else:
        base_url = 'https://api.blockchair.com/'
        first = requests.get(base_url + dlt +'/blocks?q=time('+ str_start +')&s=id(asc)').json()
        last = requests.get(base_url + dlt +'/blocks?q=time('+ str_end +')&s=id(desc)').json()
        try:
            return first['data'][0]['id'], last['data'][0]['id']
        except IndexError:
            sys.exit('\nError! There are no blocks for the given time interval, try a different one.')
     

def gz_dump(str_start, str_end, dlt):
    base_url = 'https://gz.blockchair.com/'+ dlt +'/blocks/blockchair_'+ dlt +'_blocks_'
    dates , blocks = [], []
    dates.append(str_start.replace('-', ''))
    dates.append(str_end.replace('-', ''))
    for date in dates:
        r = requests.get(base_url +''+ date +'.tsv.gz', stream=True)
        if r.status_code == API_OK:
            data = io.BytesIO(r.content)
            with gzip.GzipFile(fileobj=data) as file_in:
                block_idx = pd.Series.tolist(pd.read_csv(file_in, sep='\t', usecols=['id']))
                blocks.append([block_idx[0][0], block_idx[-1][0]])
            file_in.close()
        else:
            sys.exit(str(r.status_code) +' '+ str(r.reason))  
    return blocks[0][0], blocks[-1][0]


def tx_from_sochain(index, dlt, session, weight=False):
    base_url = 'https://chain.so/api/v2/get_block/'
    tx_url = 'https://sochain.com/api/v2/get_tx/'
    node = set()
    tx_info = []
    completed = False
    while not completed:
        r = session.get(base_url + dlt +'/'+ str(index))
        if r.status_code == API_RPS_LIMIT:
            sleep(30)
        elif r.status_code == API_OK:
            r = r.json()
            txs = r['data']['txs']
            tx_iterator = 0

            while tx_iterator < len(txs):
                tx = txs[tx_iterator]
                r = session.get(tx_url + dlt +'/'+ tx)
                if r.status_code == API_RPS_LIMIT:
                    sleep(30)
                elif r.status_code == API_OK:
                    r = r.json()
                    inputs = r['data']['inputs']
                    outputs = r['data']['outputs']
                    amount = 0
                    if weight:
                        for item in inputs:
                            amount += float(item['value'])
                    for tx_in, tx_out in product(inputs, outputs):
                        node.add(tx_in['address'])
                        node.add(tx_out['address'])
                        transaction = Transaction(tx_in['address'], tx_out['address'], amount)
                        tx_info.append(transaction)

                    tx_iterator += 1
                else:

                    tx_iterator += 1
            completed = True
        else:
            completed = True
    return node, tx_info

def tx_from_etherscan(index, session, weight=False):
    base_url = 'https://api.etherscan.io/api?module=proxy&action=eth_getBlockByNumber&tag='
    msg = 'Max rate limit reached'
    node = set()
    tx_info = []
    completed = False
    while not completed:
        r = session.get(base_url + hex(index) +'&boolean=true&apikey='+ os.getenv("eth_key"))
        r = r.json()
        if msg in r['result']:
            sleep(1)
        else:
            txs = r['result']['transactions']
            for tx in txs:      
                sender = tx['from']
                receiver = tx['to']
                amount = 0
                if weight:
                    amount = int(tx['value'], 16)
                node.add(sender)
                node.add(receiver)
                tx_info.append(Transaction(sender, receiver, amount))
            completed = True
    return node, tx_info            
