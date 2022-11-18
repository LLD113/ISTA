
def usr_info(usr_name, usr_pwd):
    filename = "UsernameAndPassword.txt"
    user_dict = {}
    try:
        with open(filename, "r") as dict_file:
            for line in dict_file:
                (username, password) = line.split(":")
                user_dict[username] = password
    except IOError as ioerror:
        print("{file}not exit".format(file = filename))
    name = usr_name
    _password = usr_pwd
    flag = False
    if name in user_dict.keys():

        if _password == user_dict[name]:
            flag = True
            return flag
    else:
        print("username not exit!")
        return flag
