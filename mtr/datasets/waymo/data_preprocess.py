# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import sys, os
import numpy as np
import pickle
import tensorflow as tf
import multiprocessing
import glob
import shutil
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type


def decode_tracks_from_proto(tracks):
    track_infos = {
        'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        'object_type': [],
        'trajs': []
    }
    for cur_data in tracks:  # number of objects
        cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                              x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos['object_id'].append(cur_data.id)
        track_infos['object_type'].append(object_type[cur_data.object_type])
        track_infos['trajs'].append(cur_traj)

    track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': []
    }
    polylines = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = lane_type[
                cur_data.lane.type]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary'] = [{
                'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                'feature_id': x.boundary_feature_id,
                'boundary_type': x.boundary_type  # roadline type
            } for x in cur_data.lane.left_boundaries
            ]
            cur_info['right_boundary'] = [{
                'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                'feature_id': x.boundary_feature_id,
                'boundary_type': road_line_type[x.boundary_type]  # roadline type
            } for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['lane'].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['speed_bump'].append(cur_info)

        else:
            pass

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print('Empty polylines: ')
    map_infos['all_polylines'] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        dynamic_map_infos['state'].append(np.array([state]))
        dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    return dynamic_map_infos


def parse_data(input_path, output_path, pre_fix=None, max_files=None):
    cnt = 0
    MAX = 100000
    ret_infos = []
    file_list = os.listdir(input_path)

    #file_list = [f for f in file_list if f.endswith('tfrecord')]

    if max_files:
        file_list = file_list[:max_files]

    for file in tqdm(file_list):
        file_path = os.path.join(input_path, file)

        dataset = tf.data.TFRecordDataset(file_path, compression_type='')

        for j, data in enumerate(dataset):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytearray(data.numpy()))

            info = {}
            info['scenario_id'] = scenario.scenario_id
            info['timestamps_seconds'] = list(scenario.timestamps_seconds)
            info['current_time_index'] = scenario.current_time_index
            info['sdc_track_index'] = scenario.sdc_track_index
            info['objects_of_interest'] = list(scenario.objects_of_interest)
            info['tracks_to_predict'] = {
                'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
                'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
            }

            track_infos = decode_tracks_from_proto(scenario.tracks)
            info['tracks_to_predict']['object_type'] = [
                track_infos['object_type'][cur_idx] for cur_idx in info['tracks_to_predict']['track_index']
            ]

            # Decode map-related data
            map_infos = decode_map_features_from_proto(scenario.map_features)
            dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

            save_infos = {
                'track_infos': track_infos,
                'dynamic_map_infos': dynamic_map_infos,
                'map_infos': map_infos
            }
            save_infos.update(info)

            ret_infos.append(info)
            
            if pre_fix is None:
                p = os.path.join(output_path, '{}.pkl'.format(cnt))
            else:
                p = os.path.join(output_path, '{}_{}.pkl'.format(pre_fix, cnt))

            with open(p, 'wb') as f:
                pickle.dump(save_infos, f)

            cnt += 1  


            if cnt >= MAX:
                return ret_infos

    return ret_infos

if __name__ == '__main__':
    PATH_A = sys.argv[1] 
    PATH_B = sys.argv[2] 
    PATH_C = sys.argv[3] 
    
    for i in range(10):
        os.makedirs('{}/{}'.format(PATH_C, i))

   
    for i in tqdm(range(10)):
        for j in tqdm(range(100)):
            src_file = '{}/uncompressed_scenario_training_20s_training_20s.tfrecord-{:05d}-of-01000'.format(PATH_A, j + 100 * i)
            dest_dir = '{}/{}/'.format(PATH_C, i)


            if os.path.exists(src_file):
                shutil.move(src_file, dest_dir)
            else:
                pass

    source_root = PATH_C + '/{}'
    target_dir = PATH_B

    all_infos = []


    for x in range(10):
        source_dir = source_root.format(x)
        infos = parse_data(source_dir, os.path.join(target_dir, 'processed_scenarios'), pre_fix=str(x))
        all_infos.extend(infos)

    filename = os.path.join(target_dir, 'processed_scenarios_infos.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(all_infos, f)
    print('----------------Waymo info train file is saved to %s----------------' % filename)