import sys
from cugraph import Graph


class Analysis:

    graph_properties = [
        'nodes_number',
        'edges_number',
        'avg_clustering_coefficient',
        'degree_distribution_tot',
        'degree_distribution_in',
        'degree_distribution_out'
    ]

    def __init__(self, direction=False):
        self.graph = Graph(directed=direction)
        self.__properties = {} # Store all the extra variables

    def new_property(self, **kwargs):
        for kwarg in kwargs:
            if kwarg not in self.graph_properties:
                raise KeyError(f'Got an unexpected key "{kwarg}"')
        
        self.__properties.update(kwargs)

    def get_graph(self):
        return self.graph


    def get_property(self, key):
        try:
            return self.__properties[key]
        except KeyError:
            sys.exit(f'Invalid key {key}')

    def get_properties(self):
        return self.__properties