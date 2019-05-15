import os, sys


def parse_file(in_folder, file_name, ids):
    # read data file
    readin = open(os.path.join(in_folder, file_name), 'r')

    # skip the 1st line
    the_line = readin.readline()

    the_line = readin.readline()
    while the_line:
        # delete the \n
        the_line = the_line.strip('\r\n')
        index = 0

        for cell in the_line.split(','):
            # filter the ST
            if index == 2 and cell.lower().startswith("st"):
                new_line = ''
                break

            # the features cols
            if index == 2 and cell not in ids:
                ids[cell] = set()
            elif index in (1, 2, 14, 20, 56, 58, 60):
                new_line = new_line + "," + cell
            index = index + 1

        the_line = readin.readline()


if __name__ == '__main__':
    in_folder = sys.argv[1]
    # in_folder = 'E:\\mldata\\hard-disk-2016-q1-data'
    out_name = in_folder + ".csv"

    ids = {}
    children = os.listdir(in_folder)
    for file_name in children:
        parse_file(in_folder, file_name, ids)
