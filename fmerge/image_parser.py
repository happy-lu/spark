import datetime
import threading
from threading import Lock
from utils.ex_util import *
from fmerge.file_merger import *
from fmerge.rest_client import *

update_lock = threading.Lock()
parsing_lock = threading.Lock()


class ImageParser(object):
    def __init__(self, conf, log):
        self.config = conf
        self.logger = log
        self.oper_history = []
        pic_folder = conf['image_folder']

        log.info("start reinit with folder: " + pic_folder)
        self.info_dict = self.reinit_infodict(pic_folder)
        log.info("success reinit with folder: %s, info_dict: %s" % (pic_folder, str(self.info_dict)))

    def reinit_infodict(self, pic_folder):
        info_dict = {}
        if os.path.exists(pic_folder) and os.path.isdir(pic_folder):
            children = os.listdir(pic_folder)
            for folder_name in children:
                session_folder = os.path.join(pic_folder, folder_name)
                if os.path.isdir(session_folder):
                    info_dict[folder_name] = set()
                    sc = os.listdir(session_folder)
                    for pic in sc:
                        if allowed_file(pic):
                            info_dict[folder_name].add(pic)
        else:
            os.makedirs(pic_folder)
        return info_dict

    def add_file(self, folder, filename):
        update_lock.acquire()
        if folder not in self.info_dict.keys():
            self.info_dict[folder] = set()

        self.info_dict[folder].add(filename)
        update_lock.release()

    def check_and_parse(self, folder, file_num):
        if file_num == len(self.info_dict[folder]):
            self.logger.info("all files uploaded, create a thread with folder: " + folder)
            self.create_parse_job(folder)

    def create_parse_job(self, folder):
        t = threading.Thread(target=self.parse_img_content, args=(folder,))
        t.start()

    def parse_img_content(self, folder):
        try:
            parsing_lock.acquire()
            self.add_oper_history(folder, "start")
            self.logger.info("start parse_img_content with folder: " + folder)
            image_path = os.path.join(self.config['image_folder'], folder)
            if not os.path.exists(image_path):
                self.logger.error("the image_path: %s has been deleted, directly return" % image_path)
                self.add_oper_history(folder, "jumped")
                return False

            text_path = os.path.join(self.config['text_folder'], folder)
            if not os.path.exists(text_path):
                os.makedirs(text_path)

            success = self.image_to_txt(image_path, text_path)
            if not success:
                self.logger.error("image to text error, image_path:" + image_path)
                self.add_oper_history(folder, "parse img failed")
                return False

            fm = FileMerger(self.config)
            json_dict = fm.start_merge(text_path, self.logger)
            self.logger.info("merge text success, folder name:" + text_path)

            result = send_result(self.config["result_post_url"], json_dict)
            self.logger.info("send rest success, rest result:" + result)

            del_file(image_path)
            self.logger.info("success del image folder:" + image_path)
            del_file(text_path)
            self.logger.info("success del text folder:" + text_path)
            del self.info_dict[folder]

            self.logger.info("success parse_img_content with folder: " + folder)
            self.add_oper_history(folder, "success")
            parsing_lock.release()
            return True
        except Exception as err:
            self.logger.error("parse_img_content failed,folder %s error:" % folder)
            self.logger.exception(err)
            self.add_oper_history(folder, "failed")
            return False

    def image_to_txt(self, image_path, text_path):
        return True

    def add_oper_history(self, folder, info):
        self.oper_history.append((datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), folder, info))
