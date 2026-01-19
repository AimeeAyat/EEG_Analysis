import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))

def main():
    print('Workspace root:', ROOT)
    nl_path = os.path.join(ROOT, 'EEGdata', 'sub01', 'name_label.npy')
    print('Loading', nl_path)
    name_label = np.load(nl_path, allow_pickle=True)
    print('name_label type:', type(name_label), 'shape:', getattr(name_label, 'shape', None))
    # Print first 20 entries
    print('\nFirst 20 name_label entries:')
    for i, v in enumerate(name_label[:20]):
        print(i, v)
    # Summarize all .npy files in this subject folder
    subj_dir = os.path.join(ROOT, 'EEGdata', 'sub01')
    print('\nListing .npy files in', subj_dir)
    for fn in sorted(os.listdir(subj_dir)):
        if fn.endswith('.npy'):
            p = os.path.join(subj_dir, fn)
            try:
                arr = np.load(p, allow_pickle=True)
                print(fn, '-> shape:', getattr(arr, 'shape', None), 'dtype:', getattr(arr, 'dtype', None))
            except Exception as e:
                print(fn, '-> could not load (', e, ')')
    # Quick comparison
    print('\nname_label length:', len(name_label))

    # Analyze label composition
    stripped = [s.split('_', 1)[1] if '_' in s else s for s in name_label]
    unique_instances = sorted(set(stripped))
    print('Unique image instances (count):', len(unique_instances))
    print('First 20 unique instances:', unique_instances[:20])

    # Verify process_data matches concatenated train+test ordering
    proc_path = os.path.join(subj_dir, 'process_data_1s_250Hz.npy')
    proc = np.load(proc_path, allow_pickle=True)
    print('\nprocess_data_1s_250Hz shape:', proc.shape)

    train_p = os.path.join(subj_dir, 'sub01_train_data_1s_250Hz.npy')
    test_p = os.path.join(subj_dir, 'sub01_test_data_1s_250Hz.npy')
    train = np.load(train_p, allow_pickle=True)
    test = np.load(test_p, allow_pickle=True)
    print('train raw shape:', train.shape, 'test raw shape:', test.shape)
    flat_train = train.reshape(-1, train.shape[-2], train.shape[-1])
    flat_test = test.reshape(-1, test.shape[-2], test.shape[-1])
    concat = np.concatenate([flat_train, flat_test], axis=0)
    print('flat_train length:', flat_train.shape[0], 'flat_test length:', flat_test.shape[0], 'concat length:', concat.shape[0])
    print('proc length:', proc.shape[0])
    eq = np.allclose(proc, concat)
    print('process_data equals concat(train,test)?', eq)

    # Try per-image interleaved concat: for each of the 72 images, append that image's train then test blocks
    per_image_list = []
    for i in range(train.shape[0]):
        t_i = train[i].reshape(-1, train.shape[-2], train.shape[-1])
        te_i = test[i].reshape(-1, test.shape[-2], test.shape[-1])
        per_image_list.append(t_i)
        per_image_list.append(te_i)
    interleaved = np.concatenate(per_image_list, axis=0)
    print('interleaved length:', interleaved.shape[0])
    eq2 = np.allclose(proc, interleaved)
    print('process_data equals per-image interleaved(train,test)?', eq2)

    # Try to reshape name_label into train/test groupings and compare label positions
    train_count = flat_train.shape[0]
    test_count = flat_test.shape[0]
    nl = np.array(name_label)
    labels_train_part = nl[:train_count].reshape(train.shape[0], train.shape[1], train.shape[2])
    labels_test_part = nl[train_count:train_count+test_count].reshape(test.shape[0], test.shape[1], test.shape[2])
    # Compare per-image: do labels_train_part[i] correspond to train[i] organization? We'll just show a sample
    print('\nSample label grouping for image 0:')
    print('labels_train_part[0] shape:', labels_train_part[0].shape)
    print(labels_train_part[0])
    print('labels_test_part[0] shape:', labels_test_part[0].shape)
    print(labels_test_part[0])





if __name__ == '__main__':
    main()
