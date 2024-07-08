from typing import Dict

import requests

from fractal.loaders.base_loader import Loader, LoaderType


class GraphLoaderException(Exception):
    pass


class BaseGraphLoader(Loader):
    """
    Base class for The Graph loaders.

    Each graph loader should inherit from this class. It should have the following attributes:
    - subgraph_id: The ID of the subgraph
    - root_url: The root URL of the graph
    - api_key: The API key to access

    Methods:
        _make_request: Make a request to the graph with a given query
    """
    def __init__(self, root_url: str, api_key: str, subgraph_id: str,
                 loader_type: LoaderType):
        super().__init__(loader_type=loader_type)
        self._url: str = f'{root_url}/{api_key}/subgraphs/id/{subgraph_id}'

    def _make_request(self, query: str, *args, **kwargs) -> Dict:
        """
        Make a request to The Graph

        Args:
            query (str): GraphQL query

        Returns:
            dict: Response data
        """
        response = requests.post(self._url, json={'query': query}, timeout=60)
        if response.status_code != 200:
            raise GraphLoaderException(f'Status code: {response.status_code}')
        data = response.json()
        if 'errors' in data:
            raise GraphLoaderException(data['errors'])
        return data['data']


class ArbitrumGraphLoader(BaseGraphLoader):
    """
    Graph Loader with arbitrum gateway
    """
    ROOT_URL = "https://gateway-arbitrum.network.thegraph.com/api"

    def __init__(self, api_key: str, subgraph_id: str, loader_type: LoaderType):
        super().__init__(root_url=self.ROOT_URL, api_key=api_key,
                         subgraph_id=subgraph_id, loader_type=loader_type)
