import json
import glob
import os

def merge_json_files(input_directory, output_file):
    """
    동일한 구조를 가진 여러 JSON 파일들을 하나의 파일로 병합합니다.
    
    Args:
        input_directory (str): JSON 파일들이 있는 디렉토리 경로 (예: "data/*.json")
        output_file (str): 병합된 결과를 저장할 파일 경로
    """
    # 모든 데이터를 저장할 리스트
    all_data = []
    
    # 입력 디렉토리의 모든 JSON 파일을 처리
    for file_path in glob.glob(input_directory):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # JSON 파일 읽기
                data = json.load(file)
                # 데이터가 리스트인지 확인하고 확장
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: {file_path}가 예상된 형식이 아닙니다.")
                print(f"{file_path} 처리 완료: {len(data)}개의 항목 추가")
        except json.JSONDecodeError as e:
            print(f"Error: {file_path} 파일 읽기 실패: {e}")
        except Exception as e:
            print(f"Error: {file_path} 처리 중 오류 발생: {e}")
    
    # 중복 제거 (선택적)
    unique_data = [dict(t) for t in {tuple(d.items()) for d in all_data}]
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(unique_data, file, ensure_ascii=False, indent=4)
    
    print(f"\n병합 완료:")
    print(f"- 총 처리된 항목 수: {len(all_data)}")
    print(f"- 중복 제거 후 항목 수: {len(unique_data)}")
    print(f"- 저장된 파일: {output_file}")


merge_json_files("/data/ephemeral/home/eval_data/*.json", "./eval_data.json")