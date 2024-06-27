import json

import pandas as pd
import requests

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory


class GMXV1FundingLoader(Loader):

    def __init__(self, token_address: str, loader_type: LoaderType):
        super().__init__(loader_type)
        self.token_address: str = token_address
        self._url: str = 'https://subgraph.satsuma-prod.com/3b2ced13c8d9/gmx/gmx-arbitrum-stats/api'

    def extract(self):
        query = """
        {
        fundingRates(
            first: 10000
            orderBy: timestamp
            orderDirection: desc
            where: {period: "daily", token: "%s"}
            subgraphError: allow
        ) {
            token
            timestamp
            startFundingRate
            startTimestamp
            endFundingRate
            endTimestamp
        }
        }
        """ % self.token_address.lower()
        response = requests.post(self._url, json={'query': query}, timeout=10)
        data = json.loads(response.text)
        self._data = pd.DataFrame(data['data']['fundingRates'])

    def transform(self):
        self._data['rate'] = (self._data['endFundingRate'] - self._data['startFundingRate']) / 1e6
        self._data['time'] = pd.to_datetime(self._data['timestamp'] * 1e9)

    def load(self):
        self._load(self.token_address)

    def read(self, with_run: bool = False) -> FundingHistory:
        if with_run:
            self.run()
        else:
            self._read(self.token_address)
        return FundingHistory(
            time=pd.to_datetime(self._data['time']).values,
            rates=(-1) * self._data['rate'].astype(float).values
        )
