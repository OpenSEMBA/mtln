# Import Module
import os

repeat=True
while repeat:
    option=input("""
    Select an option to read.
        1.- Cable
        2.- Bundle
        3.- Exit
    """
    )
    # Folder Path
    path = "/home/developer/Desktop/pythonProject/mtln/python/testData/cable_panel/sacamos_model/"
    match option:
        case "1":
            optname="CABLE"
        case "2":
            optname="BUNDLE"
        case "3":
            repeat=False
            break
    path=path+optname        
    # Cambia el directorio 
    os.chdir(path)

    # Devuelve el numero de lineas de cada archivo
    def filelength(file_path):
        with open(file_path, 'r') as f:
            return len(f.readlines())
        
    # Función que lee el archivo pasado por parametro y muestra por pantalla sus lineas
    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            print(f.read())

    # Opcion elegida por el usuario "CABLE o BUNDLE" pasada a minuscula
    optname=optname.lower()
    #Funcion que comprueba la carpeta de cables (archivos .cable y .cable_spec)
    def cable():
        cabledict={}
        match filelength(file_path):
            case 90:
                cabledict={"length":90,
                           "clave1":1,
                           "prueba":"hola"}
            case 54:
                cabledict={"length":54,
                           "clave1":1}
            case 20:
                cabledict={"length":20,
                           "clave1":1}
            case 15:
                cabledict={"length":15,
                           "clave1":1}
            case _:
                cabledict={"length":0}
        return cabledict

    cableoption=input(f"""
    Select an option to read.
        1.- .{optname}
        2.- .{optname}_spec
        3.- both
    """
    )
    x=1
    match cableoption:
        case "1":
            tu="."+optname
        case "2":
            tu="."+optname+"_spec"
        case "3":
            tu=""

    # iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(tu):
            print()
            print('---------------------------------------------------')
            print('                 SEPARATION                        ')
            print('---------------------------------------------------')
            print()
            file_path = f"{path}/{file}"

            # call read text file function
            # read_text_file(file_path)
            abc=cable() # Variable con el diccionario creado en la funcion cable
            # Compruebo los valores de la clave "length"
            match abc["length"]:
                case 90:
                    print("Archivo de 90")
                    #Creo un array con los valores del diccionario
                    pru=list(abc.values())
                    print(pru[2])
                case 54:
                    print("Archivo de 54")
                case 20:
                    print("Archivo de 20")
                case 15:
                    print("Archivo de 15")
                case 0:
                    print("Archivo con contenido no definido")

    