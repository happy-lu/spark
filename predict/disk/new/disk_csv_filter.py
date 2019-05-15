import os, sys


def parse_file(in_folder, out_name, file_name, serial_dict=None, model_dict=None):
    # read data file
    readin = open(os.path.join(in_folder, file_name), 'r')
    # write data file
    output = open(out_name, 'a')
    try:
        # skip the 1st line
        the_line = readin.readline()

        the_line = readin.readline()
        while the_line:
            # delete the \n
            the_line = the_line.strip('\r\n')
            index = 0
            output_line = ''
            new_line = ''
            for cell in the_line.split(','):
                # filter the ST
                if index == 2 and cell.lower().startswith("st"):
                    new_line = ''
                    break

                # the label col

                # the features cols
                if index == 0:
                    new_line = cell
                elif index == 4:
                    new_line = cell + "," + new_line
                elif index in (1, 2, 14, 20, 56, 58, 60):
                    new_line = new_line + "," + cell
                index = index + 1
            if len(new_line) > 0:
                output_line = output_line + new_line + '\n'
            output.write(output_line)
            the_line = readin.readline()
    finally:
        readin.close()


if __name__ == '__main__':
    in_folder = sys.argv[1]
    # in_folder = 'E:\\mldata\\hard-disk-2016-q1-data'
    out_name = in_folder + ".csv"

    children = os.listdir(in_folder)
    for file_name in children:
        parse_file(in_folder, out_name, file_name)

