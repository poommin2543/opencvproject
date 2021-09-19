# Importing Libraries
import serial
import time
from openpyxl import load_workbook
from datetime import datetime
import pandas as pd
da = ''
bar = ['MK-M48-L3-0001','MK-M48-L3-0002','MK-M48-L3-0003',
'MK-M48-L3-0004','MK-M48-L3-0005','MK-M48-L3-0006','MK-M48-L3-0007',
'MK-M48-L3-0008','MK-M48-L3-0009','MK-M48-L3-0010','MK-M48-L3-0011',
'MK-M48-L3-0012','MK-M48-L3-0013','MK-M48-L3-0014','MK-M48-L3-0015',
'MK-M48-L3-0016','MK-M48-L3-0017','MK-M48-L3-0018','MK-M48-L3-0019',
'MK-M48-L3-0020','MK-M48-L3-0021','MK-M48-L3-0022','MK-M48-L3-0023',
'MK-M48-L3-0024','MK-M48-L3-0025']

nubbar = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

arduino = serial.Serial(port='COM9', baudrate=9600, timeout=.1)
def write_read():
    #arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data
def nubdata():
    pd_xl_file = pd.ExcelFile("hello.xlsx")
    df = pd_xl_file.parse("Sheet1")
    dimensions = df.shape
    num = dimensions[0]

    print('datanumber : '+ str(num + 1))

def savedata(data):

    # datetime object containing current date and time
    now = datetime.now()

    # print("now =", now)

    workbook_name = 'hello.xlsx'
    wb = load_workbook(workbook_name)
    page = wb.active

    # New data to write:
    new_companies = [[now, data]]

    for info in new_companies:
        page.append(info)

    wb.save(filename=workbook_name)
while True:
   # num = input("Enter a number: ") # Taking input from user
    value = write_read()#.decode("utf-8")
    if value==b'' or value==b'DATA,DATE,TIME,' or value == '':
        pass
    else:
         value = value.decode("utf-8")
         if (da == value):
             pass
         else:
             try:
                 print("------------------------------------")
                 print(value) # printing the value
                 savedata(value)
                 nubdata()
                 index = bar.index(value[0:14])
                 nubbar[index] = nubbar[index] + 1
                 print("barcode count : ",end='')
                 print(nubbar[index])
                 print("------------------------------------")
                 print()
             except :
                 print("Error")
         da = value