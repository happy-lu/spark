import os


def parse_file(in_folder, out_folder, file_name, serial_dict=None, model_dict=None):
    # read data file
    readin = open(os.path.join(in_folder, file_name), 'r')
    # write data file
    output = open(os.path.join(out_folder, file_name[:-4] + '.txt'), 'w')
    try:
        the_line = readin.readline()
        while the_line:
            # delete the \n
            the_line = the_line.strip('\r\n')
            index = 0;
            output_line = ''
            for sub_line in the_line.split(','):
                # the label col
                if index == 4:
                    output_line = sub_line + " " + output_line
                # the features cols
                if sub_line != '' and index != 4:
                    # if index == 1:
                    the_text = ' ' + str(index) + ':' + sub_line
                    output_line = output_line + the_text
                index = index + 1
            output_line = output_line + '\n'
            output.write(output_line)
            the_line = readin.readline()
    finally:
        readin.close()


if __name__ == '__main__':
    in_folder = 'E:\\mldata\\hard-disk-2016-q1-data'
    out_folder = 'E:\\mldata\\libsvm'

    children = os.listdir(in_folder)
    for file_name in children:
        parse_file(in_folder, out_folder, file_name)
        break
