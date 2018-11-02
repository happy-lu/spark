#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from utils.logging_util import *
from fmerge.file_merger import *


class TestFileMerger(unittest.TestCase):

    def setUp(self):
        logger = get_logger("sql_area_finder_log", "DEBUG")
        src_file = 'E://ocr//text//sql//cs_1_1_1_zhanghm_1010000001_0_98.254.1.151_db.ITMUSER_A_98.1.31.232_1521'

        config = {'keep_items_count': 5}
        fm = FileMerger(config)

    def test_80_to_100(self):
        s1 = Student('Bart', 80)
        s2 = Student('Lisa', 100)
        self.assertEqual(s1.get_grade(), 'A')
        self.assertEqual(s2.get_grade(), 'A')


# 运行单元测试
if __name__ == '__main__':
    unittest.main()
