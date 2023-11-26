import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='manual', choices=['manual', 'average'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.strategy == "manual":
        vitpose_2d = np.load('data_2d_h36m_vitpose.npz', allow_pickle=True)['positions_2d'].item()
        pct_2d = np.load('data_2d_h36m_pct.npz', allow_pickle=True)['positions_2d'].item()
        merge_2d = {}
        for subject in vitpose_2d:
            for action in vitpose_2d[subject]:
                if subject not in merge_2d:
                    merge_2d[subject] = {action: [None, None, None, None]}
                else:
                    merge_2d[subject][action] = [None, None, None, None]

                for camera_idx in range(4):
                    vitpose_sequence = vitpose_2d[subject][action][camera_idx]
                    pct_sequence = pct_2d[subject][action][camera_idx]
                    
                    merged_sequence = np.copy(vitpose_sequence)
                    pct_joints = [2, 8, 10, 14]
                    merged_sequence[:, pct_joints, :] = pct_sequence[:, pct_joints, :]
                    merge_2d[subject][action][camera_idx] = merged_sequence
    elif args.strategy == "average":
        vitpose_2d = np.load('data_2d_h36m_vitpose.npz', allow_pickle=True)['positions_2d'].item()
        pct_2d = np.load('data_2d_h36m_pct.npz', allow_pickle=True)['positions_2d'].item()
        moganet_2d = np.load('data_2d_h36m_moganet.npz', allow_pickle=True)['positions_2d'].item()
        merge_2d = {}
        for subject in vitpose_2d:
            for action in vitpose_2d[subject]:
                if subject not in merge_2d:
                    merge_2d[subject] = {action: [None, None, None, None]}
                else:
                    merge_2d[subject][action] = [None, None, None, None]

                for camera_idx in range(4):
                    vitpose_sequence = vitpose_2d[subject][action][camera_idx]
                    pct_sequence = pct_2d[subject][action][camera_idx]
                    moganet_sequence = moganet_2d[subject][action][camera_idx]

                    merged_sequence = (vitpose_sequence + pct_sequence + moganet_sequence) / 3
                    merge_2d[subject][action][camera_idx] = merged_sequence
    else:
        raise Exception(f"Strategy {args.strategy} is not defined")
    

    print('Saving...')
    metadata = {
        'num_joints': 17,
        'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
    }
    np.savez_compressed(f"data_2d_h36m_merge_{args.strategy}", positions_2d=merge_2d, metadata=metadata)
    print('Done.')


if __name__ == '__main__':
    main()