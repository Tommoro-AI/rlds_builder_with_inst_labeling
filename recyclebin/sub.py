"""
.sub (microDVD) 자막 파일 파싱 테스트
"""


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

# 사용 예시:
if __name__ == "__main__":
    sub_file = "subtitle_0.sub"  # .sub 파일 경로를 지정하세요.
    subtitles_list = generate_frame_subtitles(sub_file)
    for frame, subtitle in subtitles_list:
        print(f"Frame {frame}: {subtitle}")
