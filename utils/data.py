import pickle, glob, os
import numpy as np
from collections import Counter


def load_data(train_patient = None, valid_patient = None, patient_label_dict = None, 
              file_name = "./data/wsi_problem.pkl", data_folder = "/data/tcga/512denseTumor",
              demo = None):
    """
    Explain wsi problem file:
        Handling Code
        0 := no use
        1 := use origin
        Error Code
        0 := Metastatic
        1 := Small tissue
        2 := Unable to detect tumor
        3 := Color variation
    """
    with open(file_name, "rb") as fp:
        problem_svs = pickle.load(fp)
        
    train_images, valid_images = [], []
    train_lookup, valid_lookup = {}, {}
    train_npys = []
    train_p_n, valid_p_n = [0, 0], [0, 0]

    for idx, npy in enumerate(sorted(glob.glob(os.path.join(data_folder, "*.npy")))):
        svs_name, _ = os.path.splitext(os.path.split(npy)[-1])
        patient_id = svs_name[:12]
        if svs_name in problem_svs.keys():
            handle_code, error_code = problem_svs[svs_name]
            if handle_code:
                npy = os.path.join("/data/tcga/512densenpy/", "{}.npy".format(svs_name))
            else:
                continue
                
        x_y_pairs = np.load(npy)
        for x, y in x_y_pairs:
            image_name = "{}_{}_{}".format(svs_name, x, y)
            if patient_id in train_patient:
                train_images.append(image_name)
                train_lookup[image_name] = patient_label_dict[patient_id]
            else:
                valid_images.append(image_name)
                valid_lookup[image_name] = patient_label_dict[patient_id]

        if patient_id in train_patient:
            train_npys.append(npy)
            train_p_n[patient_label_dict[patient_id]] += 1
        if patient_id in valid_patient:
            valid_p_n[patient_label_dict[patient_id]] += 1
            
        # demo
        if demo and idx > demo:
            break
            
    train_images = np.array(train_images)
    valid_images = np.array(valid_images)
    print("# train images:{}\n# valid images:{}".format(len(train_images), len(valid_images)))
    print("# train images:{}\n# valid images:{}".format(Counter(list(train_lookup.values())), \
                                          Counter(list(valid_lookup.values())) ))
    print("# train npys:(0: {}, 1: {})\n# valid npys:(0: {}, 1: {})".format(train_p_n[0], train_p_n[1], \
                                          valid_p_n[0], valid_p_n[1]))
    del train_p_n, valid_p_n 
    return train_images, valid_images, train_lookup, valid_lookup, train_npys
