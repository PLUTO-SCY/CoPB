# coding: utf-8

import hashlib
import json
import logging
import os
import pdb
import shutil
import sys
import time
import torch
import numpy as np
from tqdm import tqdm


def timeInday(time):
    try:
        h, m = time.split(':')
    except:
        h, m, _ = time.split(':')
    h = int(h)
    m = int(m)
    minutes = h*60+m
    return minutes/(24*60)


def timeSplit(time):
    time = time[1:-1]
    start, end = time.split(',')
    start = start.strip()
    end = end.strip()
    return (timeInday(start), timeInday(end))


def genDataProcess(trace, map):
    res = []
    for item in trace:
        poiid = item[2]
        poi = map.get_poi(poiid)
        xy = poi['position']
        position = (xy['x'], xy['y'])
        SEtime = timeSplit(item[1])

        res.append([SEtime, poiid, position])
    return res


def readOneGenTrace(profileindex, genIndex, map):
    f = open('../PrintTrace/Results/TrajectoryRes/trace_{}_{}.json'.format(
        profileindex, genIndex), 'r', encoding='utf-8')
    content = f.read()
    oneTrace = json.loads(content)
    # print(oneTrace)

    trace = genDataProcess(oneTrace, map)
    return trace


def readGenTraces(expIndex, sampleIndex): 

    file_path = f'../results/{expIndex}/sample-{sampleIndex}.pt'
    tensor = torch.load(file_path)
    numpy_array = tensor.cpu().transpose(1,2).numpy()
    return numpy_array


def readGenTraces2(map, folderName):  
    traces = []
    filePath = '/data1/shaochenyang/LLMBehav/GenTrace/Results/Trajectory_BestRes_Search/{}'.format(
        folderName)
    allfiles = os.listdir(filePath)

    success = 0
    for filename in tqdm(allfiles):
        try:
            f = open("/data1/shaochenyang/LLMBehav/GenTrace/Results/Trajectory_BestRes_Search/{}/".format(
                folderName) + filename, 'r', encoding='utf-8')
            content = f.read()
            oneTrace = json.loads(content)

            trace = genDataProcess(oneTrace, map)
            traces.append(trace)
            success += 1
        except:
            # print(filename)
            pass

    print("read all num: {}".format(success))
    print("actually all num: {}".format(len(allfiles)))
    return traces


def processRealTraces(data, map):
    traces = []
    for key, value in data.items():
        trace = []
        for point in value:
            try:
                trace.append([(point[0], point[1]), point[3],
                             map.lnglat2xy(point[4][0], point[4][1])])
            except:
                print(point)

        traces.append(trace)
    return traces


def readRealData():
    pass


def hash_args(*args):
    # json.dumps will keep the dict keys always sorted.
    string = json.dumps(args, sort_keys=True, default=str)  # frozenset
    return hashlib.md5(string.encode()).hexdigest()


def use_gpu(idx):
    # 0->2,3->1,1->3,2->0
    map = {0: 2, 3: 1, 1: 3, 2: 0}
    return map[idx]


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc


def get_gps(gps_file):
    with open(gps_file) as f:
        gpss = f.readlines()
    X = []
    Y = []
    for gps in gpss:
        x, y = float(gps.split()[0]), float(gps.split()[1])
        X.append(x)
        Y.append(y)
    return X, Y


def read_data_from_file(fp):
    """
    read a bunch of trajectory data from txt file
    :param fp:
    :return:
    """
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[int(t) for t in tmp]]
    return np.asarray(dat, dtype='int64')


def write_data_to_file(fp, dat):
    """Write a bunch of trajectory data to txt file.
    Parameters
    ----------
    fp : str
        file path of data
    dat : list
        list of trajs
    """
    with open(fp, 'w') as f:
        for i in range(len(dat)):
            line = [str(p) for p in dat[i]]
            line_s = ' '.join(line)
            f.write(line_s + '\n')


def read_logs_from_file(fp):
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[float(t) for t in tmp]]
    return np.asarray(dat, dtype='float')


def prep_workspace(workspace, datasets, oridata):
    """
    prepare a workspace directory
    :param workspace:
    :param oridata:
    :return:
    """
    data_path = '/data/stu/yangzeyu/trajgen'
    if not os.path.exists(data_path+'/%s/%s' % (datasets, workspace)):
        os.mkdir(data_path+'/%s/%s' % (datasets, workspace))
    if not os.path.exists(data_path+'/%s/%s/data' % (datasets, workspace)):
        os.mkdir(data_path+'/%s/%s/data' % (datasets, workspace))
    if not os.path.exists(data_path+'/%s/%s/logs' % (datasets, workspace)):
        os.mkdir(data_path+'/%s/%s/logs' % (datasets, workspace))
    if not os.path.exists(data_path+'/%s/%s/figs' % (datasets, workspace)):
        os.mkdir(data_path+'/%s/%s/figs' % (datasets, workspace))
    '''
    shutil.copy("../data/%s/real.data" %
                oridata, "../%s/%s/data/real.data" % (datasets,workspace))
    shutil.copy("../data/%s/val.data" %
                oridata, "../%s/%s/data/val.data" % (datasets,workspace))
    shutil.copy("../data/%s/test.data" %
                oridata, "../%s/%s/data/test.data" % (datasets,workspace))
    shutil.copy("../data/%s/dispre_10.data" %
                oridata, "../%s/%s/data/dispre.data" % (datasets,workspace))
    '''
    with open(data_path+'/%s/%s/logs/loss.log' % (datasets, workspace), 'w') as f:
        pass

    with open(data_path+'/%s/%s/logs/jsd.log' % (datasets, workspace), 'w') as f:
        pass


def get_workspace_logger(datasets):

    data_path = '../data'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s")
    fh = logging.FileHandler(data_path+'/%s/logs/all.log' %
                             (datasets), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
