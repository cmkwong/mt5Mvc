import requests
import pandas as pd

# upload the forex loader
class ApiController:

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
        print(f"Data being uploaded: {len(listData)}")
        return True

    def getDataframe(self, url: str, body: dict):
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
        print(f"Data being downloaded: {len(res['data'])}")
        # change to dataframe
        return pd.DataFrame.from_dict(res['data'])

    def createTable(self, url: str, schemaObj: dict):
        """
        :param tableName:
        :param colObj:
        :return:
        """
        body = {
            "schemaObj": schemaObj
        }
        r = requests.get(url, json=body)
        if r.status_code != 200:
            print(r.text)
            return False
        res = r.json()
        print(res)
        return True