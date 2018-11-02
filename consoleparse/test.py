import datetime

def test( name):
    name = name

    filename = '.'.join(name.split(".")[:-1])
    try:
        name_info = filename.split("_")
        filename = filename
        client_ip = name_info[0]
        username =name_info[1]
        account = name_info[2]
        db_name = name_info[3]
        # start_date = datetime.strptime(name_info[-1], "%Y%m%d%H%M%S").timestamp()
        print(username)
    except Exception as  e:
        filename = None
        print(e)

if __name__ == '__main__':
    test("192.168.9.23_user1_testuser_db135_20180814105824")
    test("192.168.9.23_user2_testuser_db175_20180814115824.txt")