"""
Read or write data in Google Spreadsheet
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from config import google_api_key_file


class SpreadSheet:

    def __init__(self, name):
        # use creds to create a client to interact with the Google Drive API
        self.scope = ['https://spreadsheets.google.com/feeds',
                      'https://www.googleapis.com/auth/drive']
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_key_file, self.scope)
        self.client = gspread.authorize(self.creds)

        # Find a workbook by name and open the first sheet
        # Make sure you use the right name here.
        self.sheet = self.client.open(name).sheet1
        self.row_id = 4

    def read(self):
        """
        Extract and print all of the values
        """
        list_of_hashes = self.sheet.get_all_records()
        list_of_lists = self.sheet.get_all_values()
        print(list_of_hashes)
        print(list_of_lists)

    def get_next_row_id(self):
        list_of_lists = self.sheet.get_all_values()
        if len(list_of_lists) > 4:
            self.row_id = len(list_of_lists) + 1
            return int(list_of_lists[-1][0]) + 1
        else:
            return 1

    def read_cell(self, row, col):
        return self.sheet.cell(row, col).value

    def write(self, row):
        index = self.row_id
        self.sheet.insert_row(row, index)
        self.row_id += 1


if __name__ == '__main__':
    spreadsheet_api = SpreadSheet("Tests clustering")
    spreadsheet_api.read()
    print(spreadsheet_api.get_next_row_id())
