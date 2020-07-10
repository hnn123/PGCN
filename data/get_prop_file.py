import os
import csv
import numpy as np

def get_gt_dict(gt_path):
    gt_dict = {}
    n = 0
    count = 0
    with open(gt_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if n == 0:
                n += 1
                continue
            video_id = row[0]
            if video_id not in gt_dict:
                gt_dict[video_id] = {'type': [int(row[2])], 'gt': [int(row[5]), int(row[6])]}
            else:
                gt_dict[video_id]['type'].append(int(row[2]))
                gt_dict[video_id]['gt'].append(int(row[5]))
                gt_dict[video_id]['gt'].append(int(row[6]))
            count += 1
            n += 1
    return gt_dict

def get_iou_overlap(prop, gt_list):

    prop1 = np.ones((gt_list.shape[0])) * prop[0]
    prop2 = np.ones((gt_list.shape[0])) * prop[1]
    x1 = np.maximum(prop1, gt_list[:, 0])
    x2 = np.minimum(prop2, gt_list[:, 1])
    x3 = np.minimum(prop1, gt_list[:, 0])
    x4 = np.maximum(prop2, gt_list[:, 1])
    insert = x2 - x1
    union = x4 - x3
    length = prop[1] - prop[0]
    insert[insert < 0] = 0
    iou = np.max(insert/union)
    index = np.argmax(insert)
    overlap = insert[index] / length
    return index, iou, overlap





def get_prop_list(prop_path):
    n = 0
    prop_list = []
    # prop_dict = {}
    with open(prop_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if n == 0:
                n += 1
                continue
            # if (int(float(row[0])/5), int(float(row[1])/5)) in prop_dict:
            #     continue
            # prop_dict[(int(float(row[0])/5), int(float(row[1])/5))] = 0
            prop_list.append((int(float(row[0])/5), int(float(row[1])/5), float(row[2]), float(row[3])))
    return prop_list


def get_prop_file(gt_path, prop_dir, propfilepath='prop_file.txt'):
    gt_dict = get_gt_dict(gt_path)
    propfile = open(propfilepath, 'w')
    n = 0
    for item in list(gt_dict.items())[:100]:
        print(item[1])
        tmp_prop_path = os.path.join(prop_dir, item[0] + '_0.csv')
        if not os.path.exists(tmp_prop_path):
            continue
        prop_list = get_prop_list(tmp_prop_path)
        post_prop_list = []
        propfile.write('#' + str(n) + '\n')
        gt = item[1]['gt']
        gt_list = np.array(gt, dtype=int).reshape(-1, 2)
        # propfile.write()
        for prop in prop_list:
            index, iou, overlap = get_iou_overlap(prop, gt_list)
            if iou < 0.7:
                continue
            if prop[2] < 0.7:
                continue
            print("{}, {}, {}, {}, {}, {}, {}".format(item[1]['type'][index], iou, overlap, prop[0], prop[1], prop[2], prop[3]))
        n += 1








if __name__ == '__main__':
    gt_path = "/media/chen/hzc/code/tad/gtad/data/tianchi_annos/train_annotations_part.csv"
    prop_dir = "/media/chen/hzc/code/tad/gtad/output_100_1/default/results"
    get_prop_file(gt_path, prop_dir)