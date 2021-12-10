import numpy as n
import json
import glob

path_cloud = '../shapenetcorev2_hdf5_2048'
path_image = '../shapenet55v2'

all_cloud_files = sorted(glob.glob(path_cloud+'/*id2file.json'))
# '../modelnet40_ply_hdf5_2048/test0_id2file.json'

for f in all_cloud_files:
    with open(f) as json_file:

        set_ = f.split('_')[0][:-1]
        num = f.split('_')[0][-1]
        print('set: ', set_, ' num: ', num)
        out = []
        # out_path = path_cloud +'/ply_data_'+set_+num+ '.json'
        out_path = path_cloud +'/'+set_+num+ '.json'
        print('name: ' + out_path)

        data = json.load(json_file)
        # print(data[0]) # bookshelf/bookshelf_0659.ply
        for path in data:
            clss = path.split('/')[0]
            name = path.split('/')[1][:-4]

            images = sorted(glob.glob(path_image+'/'+set_+'/model_'+name+'_*.jpg'))
            print(path_image+'/'+set_+'/model_'+name+'_*.jpg')
            images = ['/'.join(p.split('/')[-2:]) for p in images]
            print(len(images))
            out.append(images)

        with open(out_path, 'w') as outfile:
            json.dump(out, outfile)