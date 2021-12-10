import numpy as n
import json
import glob

path_cloud = '../modelnet40_ply_hdf5_2048'
# path_image = '../modelnet40_images_new_12x'
path_image = '../modelnet40v2'

all_cloud_files = sorted(glob.glob(path_cloud+'/*id2file.json'))
# '../modelnet40_ply_hdf5_2048/ply_data_test_0_id2file.json'

for f in all_cloud_files:
    with open(f) as json_file:

        set_ = f.split('_')[-3]
        num = f.split('_')[-2]
        print('set: ', set_, ' num: ', num)
        out = []
        # out_path = path_cloud +'/ply_data_'+set_+num+ '.json'
        out_path = path_cloud +'/ply_data_'+set_+num+ '_80.json'
        print('name: ' + out_path)

        data = json.load(json_file)
        # print(data[0]) # bookshelf/bookshelf_0659.ply
        for path in data:
            clss = path.split('/')[0]
            name = path.split('/')[1][:-4]
            # tmp = name.split('_')
            idx = int(name[name.rfind('_')+1:])
            print(path, name, idx, '%09d'%idx, name[:name.rfind('_')] + '_%09d'%idx)
            new_name = name[:name.rfind('_')] + '_%09d'%idx
            print(path_image+'/'+clss+'/'+set_+'/'+new_name+'*.jpg')
            images = sorted(glob.glob(path_image+'/'+clss+'/'+set_+'/'+new_name+'*.jpg'))
            images = ['/'.join(p.split('/')[-3:]) for p in images]
            print(len(images))
            out.append(images)

        with open(out_path, 'w') as outfile:
            json.dump(out, outfile)