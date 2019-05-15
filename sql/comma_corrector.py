from sql.sqlutils import *
import sql.keyword_exam as ke


def contains_keyword(last_line):
    keyword = {'(', 'JOIN', 'FROM'}
    for word in keyword:
        if last_line.find(word) != -1:
            return True
    return False


def append_comma(file_str):
    kwd = ke.get_cmds(read_file("data//sqlkeyword.txt"))
    # kwd = {"SELECT"}
    replace_dict = {}
    cmds = file_str.split(";")
    next_key_update = False
    last_line = ""
    for cmd in cmds:
        cmd_lines = cmd.strip().split("\n")
        if len(cmd_lines) > 1:
            for i, line in enumerate(cmd_lines):
                if len(line) == 0:
                    continue

                match_str = line.strip().upper()
                for each_kwd in kwd:
                    if match_str.startswith(each_kwd + " "):
                        if next_key_update and not contains_keyword(last_line):
                            replace_dict[line] = ";\n" + line
                            # next_key_update = False

                        next_key_update = True
                        break

                last_line = line

    print(replace_dict)
    result = file_str
    for k, v in replace_dict.items():
        result = result.replace(k, v)
    print(result)


if __name__ == '__main__':
    file_str = read_file("data//sel_testdata2.txt")
    append_comma(file_str)
