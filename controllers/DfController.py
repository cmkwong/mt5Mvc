import os
import pandas as pd
import numpy as np
from fpdf import FPDF

from controllers.PlotController import PlotController
from models.myUtils import dicModel, listModel

class DfController:
    """
    Data source: local, nodeJs server, python df
    """
    def __init__(self):
        self.plotController = PlotController()
        self.pdf = FPDF('L', 'mm', 'A4')

    def readAsDf(self, *, path: str='./docs/tables', filename: str='car_sales_data.csv'):
        fullPath = os.path.join(path, filename)
        if filename.endswith('.xlsx'):
            df = pd.read_excel(fullPath)
        elif filename.endswith('.csv'):
            df = pd.read_csv(fullPath)
        else:
            df = pd.DataFrame()
        print(f"read {filename} in {path}. ")
        return df

    def getPreviousIndex(self, currentIndex, df, limitReplace=None):
        idx = np.searchsorted(df.index, currentIndex)
        if limitReplace and idx == 0:
            return limitReplace
        return df.index[max(0, idx - 1)]

    def getNextIndex(self, currentIndex, df, limitReplace=None):
        idx = np.searchsorted(df.index, currentIndex)
        if limitReplace and idx == len(df) - 1:
            return limitReplace
        return df.index[min(idx + 1, len(df) - 1)]

    # combine the column
    def combineCols(self, df, cols, separator=',', newColName=''):
        colsListType = listModel.checkType(cols)
        if len(newColName) == 0:
            newColName = '-'.join(cols)
        if colsListType == str:
            sub_df = df[cols]
        else:
            sub_df = df[df.iloc[cols]]
        df[newColName] = sub_df.apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)
        return df

    # drop rows by selecting method
    def dropRows(self, df, arg, method):
        """
        arg: str, dict, int
        method: 'last'(int) / 'head'(int) / 'condition'(dict)
        """
        if method not in ['last', 'head', 'condition']:
            raise Exception('Wrong operation.')

        if method == 'head':
            """
            arg: int, that tail rows to be discard
            """
            df = df.drop(df.head(arg).index)
        elif method == 'tail':
            """
            arg: int, that tail rows to be discard
            """
            df = df.drop(df.tail(arg).index)
        elif method == 'condition':
            """
            arg: {key: value} that if matched to be discard
            """
            for field, value in arg.items():
                df = df.loc[df[field] != value]
        return df

    # change the dataframe column data type
    def changeColsType(self, df, cols, wantedType):
        """
        :param df: dataframe
        :param cols: col name, not accept index
        :param wantedType: loader type, float, int, etc
        :return:
        """
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = self.dropRows(df, arg={col: ''}, method='condition')
            df[col] = df[col].astype(wantedType)
        return df

    # preview the summary
    def summaryPdf(self, df, *, path: str= './docs/pdf', filename: str):
        """
        :param df: dataframe
        :param path: str, path to generate the pdf
        :param filename: str, pdf file name
        :return:
        """
        # set the dataframe display in floating point
        pd.set_option('display.float_format', lambda x: f'{x:.3f}')

        # create pdf
        self.pdf.add_page()
        self.pdf.set_font('Helvetica', 'B', 9)
        # self.pdf.image(name=self.plotController.df2Img(df.head(10), './docs/temp', 'head.png'))
        # self.pdf.image(name=self.plotController.df2Img(df.tail(10), './docs/temp', 'tail.png'))
        self.pdf.write_html(df.head(10).to_html())
        self.pdf.write_html(df.tail(10).to_html())
        self.pdf.add_page()

        txt = f'Dataset size: {df.shape}'
        self.pdf.cell(self.pdf.get_string_width(txt), 9, txt)
        # self.pdf.ln()

        txt = 'Missing values:'
        self.pdf.cell(self.pdf.get_string_width(txt), 9, txt)
        # self.pdf.ln()
        self.pdf.write_html(pd.DataFrame(df.isnull().sum(), columns=['Null Count']).to_html())

        txt = 'Variable ranges:'
        self.pdf.cell(self.pdf.get_string_width(txt), 9, txt)
        # self.pdf.ln()
        self.pdf.write_html(pd.DataFrame(df.describe()).to_html())

        # columns details
        self.pdf.add_page()
        colDetails = {}
        for col in df:
            unique = df[col].unique()
            # not discrete data
            if len(unique) > 50:
                colDetails[col] = ", ".join([f"{v}" for v in unique[:50]]) + " ..."
            # discrete data: further analysis
            else:
                counts = df[col].value_counts()
                counts_percent = df[col].value_counts(normalize=True) * 100
                colDetails[col] = ", ".join([f"{field}({counts[field]}-{counts_percent[field]:.1f}%)" for field in unique])
        df_colDetails = pd.DataFrame.from_dict(colDetails, orient='index', columns=['details'])
        self.pdf.write_html(df_colDetails.to_html())

        # get the correlation heat map
        self.pdf.add_page()
        self.pdf.image(name=self.plotController.plotCorrHeatMap(df, "./docs/temp", 'heatmap.png'), h=self.pdf.eph, w=self.pdf.epw)

        # output pdf
        self.pdf.output(os.path.join(path, filename), 'F')

