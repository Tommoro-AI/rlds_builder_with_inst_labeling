"""
Kisung Shin
virtualkss@tommoro.ai

h5py (hdf5)를 열어서 key들이 무엇이 있는지 출력하고, 각 key별 value들을 너무 길지 않게 출력하는 코드
"""

import h5py

def truncate_value(value, max_length=500):
    """긴 값 생략하여 출력"""
    value_str = str(value)
    return value_str if len(value_str) <= max_length else value_str[:max_length] + "..."

def explore_hdf5_group(group, group_name="", indent=0, preview_count=5):
    """
    HDF5 그룹 또는 파일 내의 구조를 탐색하고 출력하는 재귀 함수.
    """
    prefix = "    " * indent  # 들여쓰기
    
    for key in group.keys():
        obj = group[key]
        full_key = f"{group_name}/{key}" if group_name else key  # 전체 경로
        
        if isinstance(obj, h5py.Dataset):
            print(f"{prefix}Dataset: '{full_key}' - Shape: {obj.shape}, Dtype: {obj.dtype}")

            # 데이터 샘플링하여 출력 (값이 너무 크면 일부만 출력)
            try:
                if obj.shape:  # 다차원 배열이라면 일부만 출력
                    sample_values = obj[:preview_count]
                else:  # 단일 값
                    sample_values = obj[...]

                print(f"{prefix}Sample values: {truncate_value(sample_values)}")
            except Exception as e:
                print(f"{prefix}Error reading dataset '{full_key}': {e}")

        elif isinstance(obj, h5py.Group):
            print(f"{prefix}Group: '{full_key}' (contains sub-groups or datasets)")
            explore_hdf5_group(obj, full_key, indent + 1, preview_count)  # 재귀 탐색

def explore_hdf5(file_path, preview_count=5):
    """
    HDF5 파일을 열어서 key 목록을 출력하고, 각 key별 value를 너무 길지 않게 출력하는 함수.
    """
    with h5py.File(file_path, 'r') as file:
        print("=== HDF5 File Structure ===")
        explore_hdf5_group(file, indent=0, preview_count=preview_count)
        print("=" * 80)

# 사용 예시
explore_hdf5('/home/work/.jinupahk/virtualkss/rlds_builder_with_inst_labeling/hri_dataset/data/train/episode_0.hdf5')
