from flask import Flask, request, jsonify
from utils.ex_util import *
from werkzeug.utils import secure_filename
from fmerge.file_merger import *
from fmerge.config_reader import ConfigReader
from fmerge.image_parser import ImageParser
import os
import threading
import time
import copy

app = Flask(__name__)
global conf
global image_parser

IMAGE_FOLDER = "image_folder"


@app.route('/')
def help():
    return jsonify(
        {
            'upload image file, only support: ' + str(
                ALLOWED_EXTENSIONS): '/upload_img?s3path=/xxx/1.jpg&num_in_folder=2',
            'show info dict': '/show_info_dict',
            'show oper history': '/show_oper_history',
            'trigger image parser': '/trigger_parser?folder_name=xxx'
        })


@app.route('/show_info_dict', methods=['GET'])
def show_info_dict():
    return str(image_parser.info_dict)


@app.route('/show_oper_history', methods=['GET'])
def show_oper_history():
    return str(image_parser.oper_history)


@app.route('/trigger_parser', methods=['GET', 'POST'])
def trigger_image_parser():
    app.logger.info("trigger_image_parser begin")
    f_name = request.args['folder_name']
    if not f_name:
        err_str = "bad request, no folder_name parameter"
        app.logger.error(err_str)
        return "failed, " + err_str
    full_name = os.path.join(conf[IMAGE_FOLDER], f_name)
    if os.path.exists(full_name):
        err_str = "wrong folder_name:" + f_name
        app.logger.error(err_str)
        return "failed, " + err_str

    app.logger.info("trigger_image_parser by rest begin, folder: " + full_name)
    ip = ImageParser(conf, app.logger)
    success = ip.parse_img_content(f_name)
    app.logger.info("trigger_image_parser end")
    return "success" if success else "failed, parse_img_content error"


@app.route('/upload_img', methods=['POST'])
def upload():
    app.logger.info("upload begin")

    s3path = request.args['s3path']
    file_num = request.args['num_in_folder']
    if not s3path or not file_num:
        err_str = "bad request, no s3path or no num_in_folder parameter"
        app.logger.error(err_str)
        return "failed, " + err_str

    file_dir = conf[IMAGE_FOLDER]
    folder_name = s3path.rsplit('/', 2)[1]
    folder_path = os.path.join(file_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    upload_file = request.files['file']
    if upload_file and allowed_file(upload_file.filename):
        try:
            filename = secure_filename(upload_file.filename)
            full_path = os.path.join(folder_path, filename)
            upload_file.save(full_path)
            app.logger.info("success save the file to " + full_path)

            image_parser.add_file(folder_name, filename)
            image_parser.check_and_parse(folder_name, int(file_num))
            return 'success'
        except Exception as err:
            app.logger.error("receive image failed with file: " + filename + ", error:")
            app.logger.exception(err)
            return "failed, unknown exception, with file: " + filename
    else:
        err_str = "bad request, no file in post body"
        app.logger.error(err_str)
        return "failed, " + err_str
    app.logger.info("upload end")


def image_folder_daemon(config):
    # daemon method, check by interval
    last_dict = deepcopy(image_parser.info_dict)
    while True:
        interval = config['image_folder_check_interval']
        time.sleep(int(interval))
        cur_dict = image_parser.info_dict
        # check each folder's file count, if same with last interval, force parse it
        for folder_name, files in cur_dict.items():
            if folder_name in last_dict.keys() and len(files) == len(last_dict[folder_name]):
                app.logger.warn("folder %s didn't change in %s(s), force parse it" % (folder_name, interval))
                image_parser.create_parse_job(folder_name)

        app.logger.info("current info_dict:" + str(cur_dict))
        last_dict = deepcopy(cur_dict)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("please add the conf file, like： python xxx.py rest_svr.conf")
        exit(1)

    cp = ConfigReader(sys.argv[1])
    conf = cp.get_all_config_dict()

    app.logger = get_logger("rest_server_log", cp.get_config("log_level"))
    app.logger.info("Starting cache server on " + cp.get_config("service_port"))

    image_parser = ImageParser(conf, app.logger)
    t = threading.Thread(target=image_folder_daemon, args=(conf,))
    t.start()

    app.run(
        host='0.0.0.0',
        port=cp.get_config("service_port"),
        debug=cp.get_config("log_level") == 'DEBUG',
        threaded=True,
        processes=1
    )
