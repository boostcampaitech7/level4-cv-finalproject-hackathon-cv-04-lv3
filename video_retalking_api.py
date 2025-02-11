from fastapi import FastAPI
import uvicorn
import subprocess
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
new_env = os.environ.copy()
submodule_paths = [
    os.path.join(base_dir, "submodules/CosyVoice"),
    os.path.join(base_dir, "submodules/video-retalking"),
    os.path.join(base_dir, "submodules/video-retalking/third_part"),
    os.path.join(base_dir, "submodules/video-retalking/third_part/face_parse"),  # 추가
]
new_env["PYTHONPATH"] = ":".join(submodule_paths + [new_env.get("PYTHONPATH", "")])


app = FastAPI()

@app.post("/process/")
async def process_data(payload: dict):
    output_files = payload.get("output_files", [])

    for item in output_files:
        video_path = item["video_path"] 
        audio_path = item["audio_path"]

        conda_python = "/data/ephemeral/home/anaconda3/envs/video_retalking/bin/python"
        command = [
            conda_python,
            "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/submodules/video-retalking/inference.py",
            "--face", "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/cosyvoice_result/face/temp_video_24700.mp4",
            "--audio", "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/cosyvoice_result/audio/output_24700.wav",
            "--outfile", "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-04-lv3/retalking_result/result.mp4"
        ]

        print(f"Running command: {' '.join(command)}")  # 실행되는 명령어 출력

        return_code = subprocess.call(command, env=new_env)

        if return_code != 0:
            print(f"Error: Command exited with status {return_code}")

    return {"status": "processed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30981)
