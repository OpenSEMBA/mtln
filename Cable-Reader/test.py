from testreader import Reader
import os
t="/home/developer/Desktop/pythonProject/mtln/python/testData/cable_panel/sacamos_model/"
test=Reader(t)
test.setFolder("CABLE")
for file in os.listdir():
    print()
    print('---------------------------------------------------')
    print('                 SEPARATION                        ')
    print('---------------------------------------------------')
    print()
    type=test.cabletype(file)
    #print(type)
    # test.read_text_file(file)
    dict= test.cabledictionary(type,file)
    print(dict["type"])
    print(dict["length"])
    print
    # print(dict("type"))
    # rint(test.test(file)[1])
    #print(len(type))