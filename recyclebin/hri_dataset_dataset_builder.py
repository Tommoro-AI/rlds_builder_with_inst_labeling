"""
.ass 자막일 경우
"""

from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import h5py
import re

def time_to_seconds(t):
    """
    "H:MM:SS.CC" 형식의 문자열을 초 단위의 float 값으로 변환합니다.
    """
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_ass(filename, frame_rate=30):
    """
    .ass 파일을 파싱하여 각 프레임에 해당하는 자막을 반환합니다.
    반환형식: [(frame_idx, 'subtitle'), (frame_idx, 'subtitle'), ...]
    """
    events = []      # (start_frame, end_frame, text) 형태로 저장
    max_frame = 0    # 전체 영상의 최대 프레임 번호

    with open(filename, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            # "Dialogue:" 로 시작하는 라인만 처리
            if line.startswith("Dialogue:"):
                # "Dialogue:" 이후의 내용을 가져오고, 콤마로 9번만 분리하여 10개 필드를 얻음
                parts = line[len("Dialogue:"):].strip().split(",", 9)
                if len(parts) < 10:
                    continue  # 형식이 맞지 않는 경우 건너뛰기
                start_time = parts[1].strip()  # 두번째 필드: 시작 시간
                end_time = parts[2].strip()    # 세번째 필드: 종료 시간
                text = parts[9].strip()        # 마지막 필드: 자막 텍스트
                
                # 시간을 초 단위로 변환한 후, 프레임 번호로 변경
                start_frame = int(time_to_seconds(start_time) * frame_rate)
                end_frame = int(time_to_seconds(end_time) * frame_rate)
                events.append((start_frame, end_frame, text))
                
                if end_frame > max_frame:
                    max_frame = end_frame

    # 전체 프레임에 대해 기본 자막은 빈 문자열로 초기화
    frame_subtitles = [""] * (max_frame + 1)
    
    # 각 이벤트별로 해당 프레임 범위에 자막을 할당
    for start_frame, end_frame, text in events:
        for frame in range(start_frame, end_frame + 1):
            # 여러 자막이 겹치는 경우 공백으로 구분하여 연결
            if frame_subtitles[frame]:
                frame_subtitles[frame] += " " + text
            else:
                frame_subtitles[frame] = text

    # 결과 리스트 생성: (frame_idx, subtitle)
    result = [(idx, subtitle) for idx, subtitle in enumerate(frame_subtitles)]
    return result

def extract_value(path):
    # 정규표현식을 사용하여 episode_와 .hdf5 사이의 값을 추출합니다.
    pattern = r"data/train/episode_(.+?)\.hdf5"
    match = re.search(pattern, path)
    if match:
        return match.group(1)
    return None

class HRIDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Human-Robot Interaction dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [6x joint values, '
                            '1x gripper values].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            with h5py.File(episode_path, 'r') as data:
                action_data = data['action']
                image_data = data['observations']['images']['wrist']
                lang_inst_data = f'data/train/subtitle_{str(extract_value(episode_path))}.ass'

                # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                episode = []
                print('go!\n')
                for i, step in enumerate(action_data):
                    if i == (len(action_data) - 1):
                        language = parse_ass(lang_inst_data)[i-1][1]
                    else:
                        language = parse_ass(lang_inst_data)[i][1]
                    # compute Kona language embedding
                    language_embedding = self._embed([language])[0].numpy()

                    episode.append({
                        'observation': {
                            'image': image_data[i],
                        },
                        'action': step,
                        'discount': 1.0,
                        'reward': float(i == (len(action_data) - 1)),
                        'is_first': i == 0,
                        'is_last': i == (len(action_data) - 1),
                        'is_terminal': i == (len(action_data) - 1),
                        'language_instruction': language,
                        'language_embedding': language_embedding,
                    })

                # create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': episode_path
                    }
                }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

