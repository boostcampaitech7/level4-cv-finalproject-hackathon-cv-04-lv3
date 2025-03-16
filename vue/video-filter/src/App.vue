<template>
  <div class="app">
    <LoadingOverlay v-if="isLoading" :LoadingText="LoadingText" />
    <div v-if="!videoFile" class="upload-container" @dragover.prevent @drop="handleDrop" @click="triggerFileUpload">
      <input type="file" @change="handleFileUpload" accept="video/*" class="upload-style" ref="fileInput" />
      <div class="image-container">
        <img class="image-icon" alt="" src="@/assets/image.png" />
        <img class="mic-icon" alt="" src="@/assets/mic.png" />
        <img class="play-icon" alt="" src="@/assets/video.png" />
      </div>
      <div class="upload-text">클릭 혹은 파일을 이곳에 드롭하세요.</div>
      <div class="file-types">*.mp4, *.avi, *.wav, *.mp3</div>
    </div>
    <div v-else class="content-container">
      <div class="left-container">
        <Video ref="videoComponent" :videoUrl="videoUrl" :key="videoUrl" />
        <div class="button-container">
          <img v-if="videoFile" @click="convertToText" src="@/assets/change_button.png" alt="변환 버튼" class="change-button" />
          <img v-if="videoFile" @click="reset" src="@/assets/reset-button.png" alt="초기화 버튼" class="reset-button" />
        </div>
        <Script :transcript="transcript" @sentence-clicked="handleSentenceClick" />
      </div>
      <div class="right-container">
        <RevisedScript :transcript="revised_transcript" @update-script="handleUpdateScript" />
        <div class="button-container">
          <img @click="GenerateVoice" src="@/assets/noran.png" class="generate-button" />
          </div> 
      </div>
    </div>
  </div>
</template>

<script>
import Video from "./components/Video.vue";
import Script from "./components/Script.vue";
import RevisedScript from "./components/RevisedScript.vue";
import { processReward, processSTT, processSolar, sound_transfer } from "./utils/api";
import LoadingOverlay from "./components/LoadingOverlay.vue";

export default {
  components: {
    Video,
    Script,
    RevisedScript,
    LoadingOverlay,
  },
  data() {
    return {
      videoFile: null,
      videoUrl: null,
      transcript: [],
      revised_transcript: [],
      isLoading: false,
      LoadingText: "",
    };
  },
  methods: {
    async GenerateVoice() {
      this.isLoading = true;
      this.LoadingText = "목소리 생성 중";
      const choice_script = [];
      for (let sentence of this.transcript) {
          if (sentence.choice === 'O') {
              const revisedSentence = this.revised_transcript.find(
                  rs => rs.start === sentence.start
              );
              choice_script.push({
                  start: sentence.start,
                  end: sentence.end,
                  isModified: 0,
                  choice: sentence.choice,
                  origin_text: revisedSentence?.origin_text || sentence.text,
                  change_text: revisedSentence?.new_text || sentence.text
              });
          }
      }
      
      if (choice_script.length === 0) {
          console.warn("선택된 문장이 없습니다.");
          return;
      }
      this.LoadingText = "목소리 변환 중";
      const videoUrl = await sound_transfer(this.videoFile, choice_script);
      if (videoUrl) {
          this.videoUrl = videoUrl; // 새로운 비디오 URL로 업데이트
      }
      this.isLoading = false;
    },
    handleFileUpload(event) {
      const file = event.target.files[0];
      if (file) {
        this.videoFile = file;
        this.videoUrl = URL.createObjectURL(file);
      }
    },
    handleDrop(event) {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (file) {
        this.handleFileUpload({ target: { files: [file] } });
      }
    },
    triggerFileUpload() {
      this.$refs.fileInput.click();
    },
    async convertToText() {
      if (!this.videoFile) return;
      this.isLoading = true;

      this.LoadingText = "STT 변환 중";
      let scriptData = await processSTT(this.videoFile);

      if (!scriptData) {
        this.isLoading = false;
        alert("STT 변환에 실패했습니다.");
        return;
      }
      console.log("STT 결과 보자")
      console.log(scriptData)

      this.LoadingText = "민감발언 탐지 중";
      let revisedData = await processSolar(scriptData);
      console.log("Solar 출력")
      console.log(revisedData)
      this.LoadingText = "Solar 답변 평가 중";
      let RewardData = await processReward(revisedData);
      const origin_text_reward = RewardData[0]
      const new_text_reward = RewardData[1]
      console.log(RewardData)

      if (revisedData.length !== origin_text_reward.length || revisedData.length !== new_text_reward.length) {
          console.error("revisedData와 RewardData의 길이가 다릅니다!");
      } else {
          console.log("배열 길이 확인 완료:", revisedData.length);
      }
      let filteredData = [];
      let filteredRewards = [];
      revisedData.forEach((item, index) => {
          const rewardSum = origin_text_reward[index][0] + origin_text_reward[index][1] + origin_text_reward[index][2];
          if (rewardSum > 2.0) {  // 특정 조건 만족 시만 유지
              filteredData.push(item);
              filteredRewards.push(new_text_reward[index]);  // new_text_reward도 함께 필터링
          }
      });
      const startGroups = new Map();
      console.log("배열 1차 필터링 확인 완료:", filteredData.length);

      filteredData.forEach((item, index) => {
          const startKey = item.start;
          const rewardSum = filteredRewards[index][0] + filteredRewards[index][1] + filteredRewards[index][2];

          if (!startGroups.has(startKey)) {
              startGroups.set(startKey, { item, rewardSum });
          } else {
              if (rewardSum < startGroups.get(startKey).rewardSum) {
                  startGroups.set(startKey, { item, rewardSum });
              }
          }
      });

      revisedData = Array.from(startGroups.values()).map(entry => entry.item);
      

      console.log("필터링 후 revisedData:", revisedData);
      console.log("배열 필터링 확인 완료:", revisedData.length);
      
      scriptData = scriptData.map((sentence) => ({
        ...sentence,
        choiceStart: 0,
        choiceEnd: 0,
        choice: null, // 초기값을 null로 설정
      }));

      let scriptIndex = 0;
      for (let sentence of revisedData) {
        while (scriptIndex < scriptData.length) {
          if (scriptData[scriptIndex].start === sentence.start) {
            scriptData[scriptIndex].isModified += 1;
            break;
          }
          scriptIndex++;
        }

        sentence.thumbnail = await this.generateThumbnail(sentence.start);
      }

      this.transcript = scriptData;
      this.revised_transcript = revisedData;

      // 교정된 스크립트에 포함된 텍스트를 빨간색으로 변경
//      this.revised_transcript.forEach(sentence => {
  //      this.handleUpdateScript({ sentence, choice: 'X' });
    //  });

      this.isLoading = false;
    },
    handleSentenceClick(startTime) {
      if (this.$refs.videoComponent) {
        this.$refs.videoComponent.seekToTime(startTime);
      }
    },
    handleUpdateScript({ sentence, choice }) {
      const index = this.transcript.findIndex(s => s.start === sentence.start);
      if (index !== -1) {
        const originalText = this.transcript[index].text;
        const targetText = choice === 'O' ? sentence.origin_text : sentence.new_text;
        const replacementText = choice === 'O' ? sentence.new_text : sentence.origin_text;

        // 변경된 부분의 시작 인덱스 찾기
        const choiceStart = originalText.indexOf(targetText);
        const choiceEnd = choiceStart + replacementText.length;

        if (choiceStart !== -1) {
          this.transcript[index].text = originalText.replace(targetText, replacementText);
          this.transcript[index].choice = choice;
          this.transcript[index].choiceStart = choiceStart;
          this.transcript[index].choiceEnd = choiceEnd;
        }
      }
    },
    reset() {
      this.videoFile = null;
      this.videoUrl = null;
      this.transcript = [];
      this.revised_transcript = [];
    },
    generateThumbnail(time) {
      return new Promise((resolve) => {
        const video = document.createElement("video");
        video.src = this.videoUrl;
        video.crossOrigin = "anonymous"; // CORS 문제 방지
        video.currentTime = time;

        video.addEventListener("loadeddata", () => {
          const canvas = document.createElement("canvas");
          canvas.width = video.videoWidth / 3;
          canvas.height = video.videoHeight / 3;
          const ctx = canvas.getContext("2d");

          video.addEventListener("seeked", () => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            resolve(canvas.toDataURL("image/png"));
          });

          video.currentTime = time;
        });
      });
    },
  }
};
</script>

<style scoped>
  .app {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100vh;
    padding: 20px;
    box-sizing: border-box;
    justify-content: center;
    align-items: center;
    background-color: #F2F2F2;
  }

  .upload-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 2px dashed #ccc;
    padding: 40px;
    border-radius: 10px;
    cursor: pointer;
    text-align: center;
    width: 50%;
    height: 50%;
  }

  .upload-style {
    display: none;
  }

  .image-container {
    display: flex;
    gap: 20px;
  }

  .image-icon,
  .mic-icon,
  .play-icon {
    width: 48px;
    height: 48px;
  }

  .upload-text {
    margin-top: 20px;
    font-size: 18px;
    color: #666;
  }

  .file-types {
    margin-top: 10px;
    font-size: 14px;
    color: #999;
  }

  .content-container {
    display: flex;
    width: 100%;
    height: 100%;
  }

  .left-container {
    display: flex;
    flex-direction: column;
    width: 50%;
    align-items: center;
    gap: 15px;
    position: sticky;
    top: 20px; /* 원하는 위치에 고정 */
  }

  .right-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    width: 50%;
    padding: 20px;
    box-sizing: border-box;
  }

  .button-container {
  gap: 10px;
}

.button-container img {
  height: 50px; /* 원하는 크기로 조정 */
  width: auto; /* 비율 유지 */
}

.change-button,
.reset-button,
.generate-button {
  cursor: pointer;
}

  .change-button,
  .reset-button {
    cursor: pointer;
    width: 150px; /* 버튼 크기 조정 */
    height: auto;
  }

</style>