"""
Kisung Shin
virtualkss@tommoro.ai

tfrecord를 열어서 key들이 무엇이 있는지 출력하고, 각 key별 value들을 너무 길지 않게 출력하는 코드
"""

import tensorflow as tf

# TFRecord 파일 경로 (파일명을 적절히 변경하세요)
tfrecord_file = "/home/work/tensorflow_datasets/hri_dataset/1.0.0/hri_dataset-train.tfrecord-00000-of-00001"

# TFRecordDataset을 사용하여 파일을 읽기
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# 긴 값 생략을 위한 함수
def truncate_value(value, max_length=3000):
    value_str = str(value)
    return value_str if len(value_str) <= max_length else value_str[:max_length] + "..."

# Key 목록을 출력하기 위한 플래그
printed_keys = False  

# 데이터 출력
for raw_record in raw_dataset.take(5):  # 처음 5개만 출력
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    # 첫 번째 레코드에서 key 목록 출력
    if not printed_keys:
        keys = example.features.feature.keys()
        print("TFRecord Keys:", list(keys))
        printed_keys = True

    # Key별로 value 출력 (긴 값 생략)
    for key, feature in example.features.feature.items():
        value = feature.ListFields()  # TensorFlow에서 값 읽기
        print(f"{key}: {truncate_value(value)}")
    
    print("=" * 80)  # 구분선 추가
