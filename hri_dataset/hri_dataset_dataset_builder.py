"""
Kisung Shin
virtualkss@tommoro.ai

Human Robot Interaction dataset에 대한 RLDS 데이터셋 구축
"""

from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import h5py
import re

def generate_frame_subtitles(sub_file_path):
    """
    MicroDVD 형식의 .sub 파일을 읽어, 각 프레임에 대해 자막을 할당합니다.
    파일의 각 줄은 아래와 같은 형식이어야 합니다.
      {start_frame}{end_frame}자막내용
    예시: {0}{25}Hello World!
    
    반환 결과:
      [(0, 'Hello World!'), (1, 'Hello World!'), ..., (N, '자막내용' or '')]
    """
    subtitle_entries = []
    max_frame = 0

    # .sub 파일을 열고 각 줄을 파싱
    with open(sub_file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 정규표현식으로 {start}{end}와 자막을 추출
            match = re.match(r'\{(\d+)\}\{(\d+)\}(.*)', line)
            if match:
                start_frame = int(match.group(1))
                end_frame   = int(match.group(2))
                text        = match.group(3).strip()
                subtitle_entries.append((start_frame, end_frame, text))
                if end_frame > max_frame:
                    max_frame = end_frame

    # 모든 프레임에 대해 기본값은 빈 문자열("")로 초기화
    frame_subtitles = ["" for _ in range(max_frame + 1)]
    
    # 각 자막 항목에 대해 해당 프레임 범위에 자막을 할당
    for start, end, text in subtitle_entries:
        # start~end 프레임 모두에 동일한 자막 할당
        for frame in range(start, end + 1):
            frame_subtitles[frame] = text

    # (frame_idx, subtitle) 형태의 리스트 생성
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
                lang_inst_data = f'data/train/subtitle_{str(extract_value(episode_path))}.sub'

                # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                episode = []
                for i, step in enumerate(action_data):
                    if i == (len(action_data) - 1): # Aegisub에서는 마지막 프레임에 자막을 할당하지 않음
                        language = generate_frame_subtitles(lang_inst_data)[i-1][1]
                    else:
                        language = generate_frame_subtitles(lang_inst_data)[i][1]
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

