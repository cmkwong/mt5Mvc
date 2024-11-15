import os
import pandas as pd
import numpy as np
from fpdf import FPDF

from mt5Mvc.models.myUtils import listModel, fileModel, timeModel, inputModel
from mt5Mvc.controllers.PlotController import PlotController

class DfController:
    """
    Data source: local, nodeJs server, python df
    """
    def __init__(self):
        self.plotController = PlotController()
        self.pdf = FPDF('L', 'mm', 'A4')

    def read_as_df(self, path: str, filename: str, chunksize=None, colnames=None, header=0):
        fullPath = os.path.join(path, filename)
        if filename.endswith('.xlsx'):
            df = pd.read_excel(fullPath)  # read_excel has no chunksize parameter
        elif filename.endswith('.csv'):
            df = pd.read_csv(fullPath, chunksize=chunksize, names=colnames, header=header)
        else:
            df = pd.DataFrame()
        print(f"read {filename} in {path}. ")
        return filename, df

    def read_as_df_with_selection(self, *, path: str= './docs/datas', chunksize=None, colnames=None, header=0):
        """
        :param path: read as dataframe
        """
        # ask user to select the file name
        fileList = fileModel.getFileList(path)
        fileNum = inputModel.askSelection(fileList)
        filename = fileList[fileNum]

        # assign user selected file
        filename, df = self.read_as_df(path, filename, chunksize, colnames, header)
        return filename, df

    def get_previous_index(self, currentIndex, df, limitReplace=None):
        idx = np.searchsorted(df.index, currentIndex)
        if limitReplace and idx == 0:
            return limitReplace
        return df.index[max(0, idx - 1)]

    def get_next_index(self, currentIndex, df, limitReplace=None):
        idx = np.searchsorted(df.index, currentIndex)
        if limitReplace and idx == len(df) - 1:
            return limitReplace
        return df.index[min(idx + 1, len(df) - 1)]

    # combine the column
    def combine_cols(self, df, cols, separator=',', newColName=''):
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
    def drop_rows(self, df, arg, method):
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
    def change_cols_type(self, df, cols, wantedType):
        """
        :param df: dataframe
        :param cols: col name, not accept index
        :param wantedType: loader type, float, int, etc
        :return:
        """
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = self.drop_rows(df, arg={col: ''}, method='condition')
            df[col] = df[col].astype(wantedType)
        return df

    # preview the summary
    def summaryPdf(self, df, *, path: str= './docs/pdf'):
        """
        :param df: dataframe
        :param path: str, path to generate the pdf
        :param filename: str, pdf file name
        :return:
        """
        # set the dataframe display in floating point
        pd.set_option('display.float_format', lambda x: f'{x:.3f}')
        self.pdf.set_font('Helvetica', 'B', 9)

        # create pdf
        self.pdf.add_page()
        self.pdf.write_html(df.head(10).to_html())
        self.pdf.write_html(df.tail(10).to_html())

        self.pdf.add_page()
        txt = f'Dataset size: {df.shape}'
        self.pdf.cell(self.pdf.get_string_width(txt), 9, txt)
        self.pdf.ln()

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
                colDetails[col] = ", ".join([f"{v}" for v in unique[:10]]) + " ..."
            # discrete data: further analysis
            else:
                counts = df[col].value_counts(dropna=False)
                counts_percent = df[col].value_counts(dropna=False, normalize=True) * 100
                colDetails[col] = ", ".join([f"{field}({counts[field]}[{counts_percent[field]:.1f}%])" for field in unique])
        df_colDetails = pd.DataFrame.from_dict(colDetails, orient='index', columns=['details'])
        self.pdf.write_html(df_colDetails.to_html())

        # get the correlation heat map
        self.pdf.add_page()
        self.pdf.image(name=self.plotController.getCorrHeatmapImg(df, "./docs/temp", 'heatmap.png'), h=self.pdf.eph, w=self.pdf.epw)

        # output pdf
        timeStr = timeModel.getTimeS(outputFormat="%Y%m%d%H%M%S")
        filename = f"{timeStr}_summary.pdf"
        pdf_fullPath = os.path.join(path, filename)
        print(f"File is output to {pdf_fullPath}")
        self.pdf.output(pdf_fullPath, 'F')

