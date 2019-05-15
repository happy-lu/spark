if __name__ == '__main__':

    # read data file
    readin = open('E:\\mldata\\disknew\\failed2.csv', 'r')
    # write data file
    output = open('E:\\mldata\\disknew\\failed2.txt', 'w')
    try:
        the_line = readin.readline()
        while the_line:
            # delete the \n
            the_line = the_line.strip('\r\n')
            index = 0;
            output_line = ''
            for sub_line in the_line.split(','):
                # the label col
                if index == 0:
                    output_line = sub_line
                # the features cols
                if sub_line != 'NULL' and index != 0:
                    the_text = ' ' + str(index) + ':' + sub_line
                    output_line = output_line + the_text
                index = index + 1
            output_line = output_line + '\n'
            output.write(output_line)
            the_line = readin.readline()
    finally:
        readin.close()
