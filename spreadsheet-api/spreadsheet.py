import gspread
from oauth2client.service_account import ServiceAccountCredentials


class SpreadSheet:

    def __init__(self, name):
        # use creds to create a client to interact with the Google Drive API
        self.scope = ['https://spreadsheets.google.com/feeds',
                      'https://www.googleapis.com/auth/drive']
        self.creds = ServiceAccountCredentials.from_json_keyfile_name('spreadsheet-api/client_secret.json', self.scope)
        self.client = gspread.authorize(self.creds)

        # Find a workbook by name and open the first sheet
        # Make sure you use the right name here.
        self.sheet = self.client.open(name).sheet1
        self.row_id = 4

    def read(self):
        # Extract and print all of the values
        list_of_hashes = self.sheet.get_all_records()
        list_of_lists = self.sheet.get_all_values()
        print(list_of_hashes)
        print(list_of_lists)

    def write(self, row):
        # self.sheet.update_cell(19, 3, "I just wrote to a spreadsheet using Python!")
        # row = ["I'm", "inserting", "a", "row", "into", "a,", "Spreadsheet", "with", "Python"]
        index = self.row_id
        self.sheet.insert_row(row, index)
        self.row_id += 1


if __name__ == '__main__':
    spreadsheet_api = SpreadSheet("Tests clustering")
    spreadsheet_api.read()
    # spreadsheet_api.write(["I'm", "inserting", "a", "row", "into", "a,", "Spreadsheet", "with", "Python"])
