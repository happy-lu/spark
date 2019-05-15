def read_file(file_path):
    with open(file_path, "r", encoding='UTF-8') as file:
        lines = file.readlines()
        full_str = "".join(lines)

    return full_str
