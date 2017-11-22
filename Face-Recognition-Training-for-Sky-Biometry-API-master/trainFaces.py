#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name: SkyBiometry Face Detection Trainer
# Description: trains the SkyBiometry API
#
# Author: Gerald Baeck (http://dev.baeck.at)
# License: Public Domain

import logging
import os

from face_client import FaceClient, FaceError

from sky_cfg import *  # config file

# logging stuff
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def main():
    log.info('Training started')
    client = FaceClient(FACE_API_KEY, FACE_API_SECRET)

    for root, dirs, files in os.walk(SAMPLES_DIR):
        person_name = root.replace("%s/" % SAMPLES_DIR, '')
        person_uid = "%s@%s" % (person_name.replace(' ', ''), FACE_NAMESPACE)
        files = filter(lambda f: not f.startswith('.'), files)
        log.debug("Looking at %s directory" % person_name)

        if len(files) > 0:
            tids = list()
            for f in files:
                if not f.startswith('.'):
                    file_name = "%s/%s" % (root, f)
                    log.info("Uploading %s" % file_name)
                    try:
                        response = client.faces_detect(file=file_name)
                        tids += [photo['tags'][0]['tid'] for photo in response['photos']]
                    except FaceError, e:
                        log.error(e)
            log.debug("Saving/Training images for %s" % person_name)

            response = client.tags_save(tids=','.join(
                tids), uid=person_uid, label=person_name)  # save the uploaded files permanently
            if 'status' in response and response['status'] == 'success':
                response = client.faces_train(person_uid)  # train the API
                if 'status' in response and response['status'] == 'success':
                    log.info("Saving/Training images for %s succeded" % person_name)
                else:
                    log.error("Training failed:\n%s" % response)
            else:
                log.error("Saving failed:\n%s" % response)

    log.info('Training finished')


if __name__ == "__main__":
    main()
