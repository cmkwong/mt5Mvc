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
        self.uploadForexDataUrl = self.mainUrl + "/forex/table/upload"
        self.downloadForexDataUrl = self.mainUrl + "/forex/table/download?tableName={}"
        self.createTableUrl = self.mainUrl + "/forex/table/create"
        self.allSymbolInfoUrl = self.mainUrl + "/forex/symbol/info"
        # get strategy param
        self.strategyParamUrl = self.mainUrl + "/forex/strategy/param"
        # update the strategy running records
        self.dealRecordUrl = self.mainUrl + "/forex/strategy/deal/record"

    # restful API format: GET / POST
    def restRequest(self, url: str, param: dict = None, body: dict = None, restType='GET'):
        if param:
            args = []
            for k, v in param.items():
                args.append(f"{k}={v}")
            url += f"?{'&'.join(args)}"
        if restType == 'GET':
            r = requests.get(url, json=body)
        elif restType == 'POST':
            r = requests.post(url, json=body)
        else:
            print(f"{restType} is not matched.")
            return False
        if r.status_code != 200:
            print(r.text)
            return False
        res = r.json()
        print(res)
        return res

    def postDataframe(self, url: str, df: pd.DataFrame, param: dict = None):
        """
        upload forex Data ohlcvs: open, high, low, close, volume, spread
        """
        df = df.fillna('')
        if len(df) == 0:
            print("No Data")
            return False
        listData = df.to_dict('records')
        self.restRequest(url, param, {'data': listData}, 'POST')
        # r = requests.post(url, json={'data': listData})
        return True

    def getDataframe(self, url: str, param: dict = None, body: dict = None):
        """
        download forex ohlcvs from server
        :param url: str
        :param body: dict
        :return pd.DataFrame with ohlcvs
        """
        res = self.restRequest(url, param, body)
        # r = requests.get(url, json=body)
        # res = r.json()
        # if r.status_code != 200:
        #     print(r.text)
        #     return False
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
