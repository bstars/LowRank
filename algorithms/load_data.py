from scipy.io import loadmat
import scipy.sparse as SP
import numpy as np
import matplotlib.pyplot as plt

def load_yale(num_obj=None):
    """
    :return: face set of shape [48, 42, 64, 38],
                38 peoples, 64 images for each people, each image of shape (48,42)
    """
    mdict = loadmat('../data/YaleBCrop025.mat')
    Y = mdict['Y']
    Y = np.reshape(Y, newshape=[42,48,64,38]) / 256
    if num_obj is not None:
        Y = Y[:,:,:,:num_obj]
    return np.transpose(Y, axes=[1,0,2,3])

def load_multi_site_yale(num_obj, sites):
    faces = load_yale(num_obj)
    Xs = [[] for i in sites]


    for i,s in enumerate(sites):
        for j in range(num_obj):
            Xs[i].append(
                np.reshape(faces[:,:,s,j], [-1])
            )
    ret = [np.stack(X).T for X in Xs]
    return ret


def load_abide_processed():
    mdict = loadmat('../data/rois_aal.mat')
    fea = mdict['fea']
    label = mdict['label']
    all_count = mdict['All_Count']
    all_count_name = mdict['All_Count_Name']
    used_num = mdict['Used_Num']

    num_sites = fea.shape[0]
    site_names = []
    datas = []
    labels = []



    idx = np.triu_indices(116,1)
    for i in range(num_sites):
        site_names.append(
            all_count_name[i,0][0]
        )
        labels.extend(
            label[i, 0][0]
        )

        fea_site = fea[i,0]

        if np.ndim(fea_site) == 3:
            num_obj_site = fea_site.shape[2]
            datas_site = []

            for j in range(num_obj_site):
                datas_site.append(
                    fea_site[:,:,j][idx]
                )
            datas.append(np.stack(datas_site).T)

        else:
            datas.append(
                np.expand_dims(fea_site[idx], 0).T
            )


    labels = np.array(labels)
    return datas, labels, site_names












if __name__ == '__main__':
    datas, labels, site_names = load_abide_processed()

    X = SP.block_diag(datas)
    print(X.shape)