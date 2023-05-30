from testreader import Reader
path="/home/repository/testData/inputs/"
reader=Reader(path)
reader.setFolder("")
file="bare_wire.cable"
dict=reader.cabledictionary(file)
#Keys introduced by my self
#keys=["cable_type_number","total_number_of_conductor","number_of_external_conductors","total_number_of domains","number_of_internal_conductors","number_of_internal_domains","number_of_cable_parameters","cable_parameters","number_of_frequency_dependent_dielectric_models","dielectric_filters","conductor_impedance_models"]
#keysDF=["number","w_normalisation_constant","a_coefficients","b_coefficients"]
#keysCIM=["conductor_impedance_model_type","conductor_radius","conductivity","resistance_multiplication_factor","external_to_domain_conductor_currrent_transformation_matrix_MI","external_to_domain_conductor_currrent_transformation_matrix_MV","local_reference_conductor_for_internal_domains","number_of_conductros_in_each_domain","external_conductor_information_and_dielectric_model"]
#keysECIDM=["conductor_type","conductor_radius","conductor_width","conductor_width2","conductor_height","conductor_ox_oy","dielectric_radius","dielectric_width","dielectric_height","dielectric_ox_oy","w_normalisation_constant","a_coefficients","b_coefficients"]

#Global dictionary from the file
keys=list(dict.keys())

#Dictionary from the file for "dielectric_filters"
dictDF=dict["dielectric_filters"]
keysDF=list(dictDF.keys())

#Dictionary from the file for "conductor_impedance_models"
dictCIM=dict["conductor_impedance_models"]
keysCIM=list(dictCIM.keys())

#Dictionary from the file for "external_conductor_information_and_dielectric_model" at "conductor_impedance_models"
dictECIDM=dict["conductor_impedance_models"]["external_conductor_information_and_dielectric_model"]
keysECIDM=list(dictECIDM.keys())

#Dictionary from the file for "conductor_labels"
dictCL=dict["conductor_labels"]
keysCL=list(dictCL.keys())

#Tests for dictionary
assert(dict[keys[0]]=="v4.0.0")
assert(dict[keys[1]]=="cylindrical")
assert(dict[keys[2]]=="1")
assert(dict[keys[3]]=="1")
assert(dict[keys[4]]=="1")
assert(dict[keys[5]]=="1")
assert(dict[keys[6]]=="0")
assert(dict[keys[7]]=="0")
assert(dict[keys[8]]=="3")
assert(dict[keys[9]]==["1.0e-04", "1.0e-04", "0.0e+00"])
assert(dict[keys[10]]=="1")
assert(dict[keys[11]]==dictDF)
assert(dict[keys[12]]==dictCIM)
assert(dict[keys[13]]==dictCL)

#Tests for dictionary "dielectric_filters" (Inside the global dictionary)
assert(dictDF[keysDF[0]]=="1")
assert(dictDF[keysDF[1]]=="1.0e+00")
assert(dictDF[keysDF[2]]=="1.0e+00")
assert(dictDF[keysDF[3]]=="1.0e+00")

#Tests for dictionary "conductor_impedance_models" (Inside the global dictionary)
assert(dictCIM[keysCIM[0]]=="1")
assert(dictCIM[keysCIM[1]]=="1.0e-04")
assert(dictCIM[keysCIM[2]]=="0.0e+00")
assert(dictCIM[keysCIM[3]]=="1.0e+00")
assert(dictCIM[keysCIM[4]]==[["1.0e+00","0.0e+00"],["1.0e+00","1.0e+00"]])
assert(dictCIM[keysCIM[5]]==[["1.0e+00","-1.0e+00"],["0.0e+00","1.0e+00"]])
assert(dictCIM[keysCIM[6]]=="0")
assert(dictCIM[keysCIM[7]]=="2")

#Tests for dictionary "external_conductor_information_and_dielectric_model" (Inside the "conductor_impedance_models" dictionary)
assert(dictECIDM[keysECIDM[0]]=="1")
assert(dictECIDM[keysECIDM[1]]=="1.0e-04")
assert(dictECIDM[keysECIDM[2]]=="0.0e+00")
assert(dictECIDM[keysECIDM[3]]=="0.0e+00")
assert(dictECIDM[keysECIDM[4]]=="0.0e+00")
assert(dictECIDM[keysECIDM[5]]==["0.0e+00","0.0e+00"])
assert(dictECIDM[keysECIDM[6]]=="1.0e-04")
assert(dictECIDM[keysECIDM[7]]=="0.0e+00")
assert(dictECIDM[keysECIDM[8]]=="0.0e+00")
assert(dictECIDM[keysECIDM[9]]==["0.0e+00","0.0e+00"])
assert(dictECIDM[keysECIDM[10]]=="1.0e+00")
assert(dictECIDM[keysECIDM[11]]=="1.0e+00")
assert(dictECIDM[keysECIDM[12]]=="1.0e+00")

#Tests for dictionary "conductor_labels" (Inside the global dictionary)
assert(dictCL[keysCL[0]]=="bare_wire")
assert(dictCL[keysCL[1]]=="cylindrical")
assert(dictCL[keysCL[2]]=="wire")

#If all the tests are correct this message is shown at the console
print("Every test is working")
