import requests
import pandas as pd


# upload the forex loader
class HttpController:

    def switchEnv(self, env):
        if env == 'prod':
            self.mainUrl = "http://192.168.1.165:3002/api/v1/query"
        else:
            self.mainUrl = "http://localhost:3002/api/v1/query"
        print(f"Connecting to {self.mainUrl} ... ")
        # define the url
        self.uploadForexDataUrl = self.mainUrl + "/forex/table/upload?tableName={}"
        self.downloadForexDataUrl = self.mainUrl + "/forex/table/download?tableName={}"
        self.createTableUrl = self.mainUrl + "/forex/table/create?tableName={}&schemaType={}"
        self.allSymbolInfoUrl = self.mainUrl + "/forex/symbol/info"
        # get strategy param
        self.strategyParamUrl = self.mainUrl + "/forex/strategy/param?{}"
        # self.backtestStrategyParamUrl = self.mainUrl + "/forex/strategy/param?name={}&live={}"

    def postDataframe(self, url: str, df: pd.DataFrame):
        """
        upload forex Data ohlcvs: open, high, low, close, volume, spread
        """
        df = df.fillna('')
        if len(df) == 0:
            print("No Data")
            return False
        listData = df.to_dict('records')
        r = requests.post(url, json={'data': listData})
        if r.status_code != 200:
            print(r.text)
            return False
        return True

    def getDataframe(self, url: str, body: dict = None):
        """
        download forex ohlcvs from server
        :param url: str
        :param body: dict
        :return pd.DataFrame with ohlcvs
        """
        r = requests.get(url, json=body)
        res = r.json()
        if r.status_code != 200:
            print(r.text)
            return False
        # change to dataframe
        return pd.DataFrame.from_dict(res['data'])

    # def createTable(self, url: str, schemaObj: dict):
    #     """
    #     :param tableName:
    #     :param colObj:
    #     :return:
    #     """
    #     body = {
    #         "schemaObj": schemaObj
    #     }
    #     r = requests.get(url, json=body)
    #     if r.status_code != 200:
    #         print(r.text)
    #         return False
    #     res = r.json()
    #     print(res)
    #     return True

    def getRequest(self, url: str, body: dict = None):
        r = requests.get(url, json=body)
        if r.status_code != 200:
            print(r.text)
            return False
        res = r.json()
        print(res)
        return True