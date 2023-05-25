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
        
    #Returns an array with the content of the selected line in parameter
    def arrayLine(self,file , number):
          with open(self.path+file, 'r') as f:
            return self.readline(file,number).split()

    #Returns the word from the line selected at the position of the array selected       
    def wordSelector(self,file,line,position):
        return self.arrayLine(file,line)[position]


    def formatNumber(self,file,line,position):
        return format(float(self.wordSelector(file,line,position)),".1e")
    
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
    def cabledictionary(self, file):
        type=self.cabletype(file)
        option=type
        if(type=="..\MOD_CABLE_PANEL\CABLE\\"):
            type=self.readline(file,2)+" CABLE SPEC"
            if(self.folder=="BUNDLE"):
                type=self.readline(file,5)+", "+self.readline(file,7)+", "+self.readline(file,9)+" BUNDLE SPEC"
        cabledict={"format":self.readline(file,0),
                    "type":type}
        match option:
            case "twisted_pair":
                cabledict.update({"f1":self.readline(file,5)})
            case "cylindrical":
                cabledict.update({"cable_type_number":self.wordSelector(file,2,0),
                                "total_number_of_conductor":self.wordSelector(file,3,0),
                                "number_of_external_conductors":self.wordSelector(file,4,0),
                                "total_number_of domains":self.wordSelector(file,5,0),
                                "number_of_internal_conductors":self.wordSelector(file,6,0),
                                "number_of_internal_domains":self.wordSelector(file,7,0),
                                "number_of_cable_parameters":self.wordSelector(file,8,0),
                                "cable_parameters":[self.formatNumber(file,9,0),
                                                    self.formatNumber(file,10,0),
                                                    self.formatNumber(file,11,0)],
                                "number_of_frequency_dependent_dielectric_models":self.wordSelector(file,12,0),
                                "dielectric_filters":{
                                    "number": self.wordSelector(file,13,4),
                                    "w_normalisation_constant":self.formatNumber(file,14,0),
                                    "a_coefficients": self.formatNumber(file,16,0),
                                    "b_coefficients": self.formatNumber(file,18,0)
                                },
                                "conductor_impedance_models":{
                                    "conductor_impedance_model_type": self.wordSelector(file,20,0),
                                    "conductor_radius": self.formatNumber(file,21,0),
                                    "conductivity": self.formatNumber(file,22,0),
                                    "resistance_multiplication_factor": self.formatNumber(file,23,0),
                                    "external_to_domain_conductor_currrent_transformation_matrix_MI":
                                    [
                                        [self.formatNumber(file,26,0), self.formatNumber(file,26,1)],
                                        [self.formatNumber(file,27,0), self.formatNumber(file,27,1)]
                                    ],
                                    "external_to_domain_conductor_currrent_transformation_matrix_MV":
                                    [
                                        [self.formatNumber(file,30,0), self.formatNumber(file,30,1)],
                                        [self.formatNumber(file,31,0), self.formatNumber(file,31,1)]
                                    ],
                                    "local_reference_conductor_for_internal_domains": self.wordSelector(file,33,0),
                                    "number_of_conductros_in_each_domain": self.wordSelector(file,35,0),
                                    "external_conductor_information_and_dielectric_model":{
                                        "conductor_type": self.wordSelector(file,37,0),
                                        "conductor_radius": self.formatNumber(file,38,0),
                                        "conductor_width": self.formatNumber(file,39,0),
                                        "conductor_width2": self.formatNumber(file,40,0),
                                        "conductor_height": self.formatNumber(file,41,0),
                                        "conductor_ox_oy": [self.formatNumber(file,42,0), self.formatNumber(file,42,1)],
                                        "dielectric_radius": self.formatNumber(file,43,0),
                                        "dielectric_width": self.formatNumber(file,44,0),
                                        "dielectric_height": self.formatNumber(file,45,0),
                                        "dielectric_ox_oy": [self.formatNumber(file,46,0), self.formatNumber(file,46,1)],
                                        "w_normalisation_constant": self.formatNumber(file,47,0),
                                        "a_coefficients": self.formatNumber(file,49,0),
                                        "b_coefficients": self.formatNumber(file,51,0)
                                    }
                                },
                                "conductor_labels":{
                                    "cable_name": self.wordSelector(file,53,2).replace('.',''),
                                    "type": self.wordSelector(file,53,4).replace('.',''),
                                    "conductor": self.wordSelector(file,53,8)
                                }
                                                                
                                })




            case "ERROR":
                cabledict.update({"length":0})
            case "..\MOD_CABLE_PANEL\CABLE\\":
                cabledict.update({"length":1})
        return cabledict
