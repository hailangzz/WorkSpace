#!/usr/bin/env bash
# -*- coding: utf-8 -*-

import logging  
import os
import sys
import time

# reload(sys)
# sys.setdefaultencoding('utf-8')

class logging:
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  
    
    rq = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    program_entrance_path = (os.path.realpath(sys.argv[0]))
    log_path = program_entrance_path.replace(filename, '') + '../Logs'

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_name = log_path +'/'+ rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='a',encoding='utf-8')
    fh.setLevel(logging.DEBUG)  
    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)

    def write_logs(self,log_info,log_type='info'):
        if log_type=='info':
            self.logger.info(log_info)
        if log_type=='warning':
            self.logger.info(log_info)
        if log_type=='error':
            self.logger.info(log_info)
