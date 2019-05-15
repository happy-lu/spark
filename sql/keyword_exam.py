from sql.sqlutils import *


def get_cmds(file_str):
    result = set()

    for line in file_str.split("\n"):
        lines = line.strip().split(" ")
        if len(lines) > 1:
            result.add(lines[0])
        elif len(lines[0]) > 0:
            result.add(lines[0])

    return result


if __name__ == '__main__':
    data = get_cmds(read_file("data//sqlkeyword.txt"))
    print(len(data))
    print(data)

    s = eval(
        """{'VALIDATE', 'ALTER', 'INSERT', 'DROP', 'REVOKE', 'COMMIT', 'DISASSOCIATE', 'SET', 'USER',
         'SELECT', 'GRANT', 'NO-OP', 'DELETE', 'ANALYZE', 'EXECUTE', 'DISABLE', 'NOAUDIT', 'EXPLAIN', 
         'AUDIT', 'LOGON', 'CALL', 'NETWORK', 'UPDATE', 'COMMENT', 'RENAME', 'LOGOFF', 'SAVEPOINT',
          'ASSOCIATE', 'CREATE', 'TRUNCATE', 'SESSION', 'PL/SQL', 'ROLLBACK', 'LOCK', 'ENABLE',
           'SYSTEM'}""")
    print(type(s))
