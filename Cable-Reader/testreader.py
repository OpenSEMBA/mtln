import os

class Reader:
    # Creating the reader
    def __init__(self,path:str):
        self.initialpath=path 

    # For setting the folder "BUNDLE" or "CABLE"
    def setFolder(self, type):
        self.path=self.initialpath+type+"/"
        self.folder=type
        os.chdir(self.path)

    # Returns the number of the files lines
    def filelength(self, file):
        with open(self.path+file, 'r') as f:
            return len(f.readlines())
        
    # Returns a new array without the Line Break
    def replaceLineBreak(self,file):
        with open(self.path+file, 'r') as f:
            filecontent=f.readlines()
            newcontent=[]
            for rep in filecontent:
                newcontent.append(rep.replace("\n",""))
            return newcontent
        
    # Returns the text from the line in the parameter
    def readline(self, file, number):
        with open(self.path+file, 'r') as f:
            return self.replaceLineBreak(file)[number]
        
    # Reads the type used for the cable    
    def cabletype(self, file):
        with open(self.path+file, 'r') as f:
            if(self.filelength(file)>2):
                value= self.readline(file,1)
            else:
                value="ERROR"
        return value
        
    # Reads and prints all the file
    def read_text_file(self, file):
        with open(self.path+file, 'r') as f:
            print(f.read())

    # Creates and returns dictionary for cables
    def cabledictionary(self,option, file):
        type=self.cabletype(file)
        if(type=="..\MOD_CABLE_PANEL\CABLE\\"):
            type=self.readline(file,2)+" CABLE SPEC"
            if(self.folder=="BUNDLE"):
                type=self.readline(file,5)+", "+self.readline(file,7)+", "+self.readline(file,9)+" BUNDLE SPEC"
        cabledict={"length":self.filelength(file),
                    "type":type}
        match option:
            case "twisted_pair":
                cabledict.update({"f1":self.readline(file,5)})
            case "cylindrical":
                cabledict.update({"f1":self.readline(file,7)})
            case "ERROR":
                cabledict.update({"length":0})
            case "..\MOD_CABLE_PANEL\CABLE\\":
                cabledict.update({"length":1})
        return cabledict
