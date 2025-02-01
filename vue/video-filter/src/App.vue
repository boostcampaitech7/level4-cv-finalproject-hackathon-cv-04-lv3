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
    <div v-else class="left-container">
      <Video ref="videoComponent" :videoUrl="videoUrl" :key="videoUrl" />
      <button v-if="videoFile" @click="convertToText">변환</button>
      <Script :transcript="transcript" @sentence-clicked="handleSentenceClick" />
    </div>
    <div v-if="videoFile" class="right-containfr">
      <RevisedScript :transcript="revised_transcript" @sentence-clicked="handleSentenceClick" />
    </div>
  </div>
</template>

<script>
import Video from "./components/Video.vue";
import Script from "./components/Script.vue";
import RevisedScript from "./components/RevisedScript.vue";
import { processSTT } from "./utils/stt.js";
import { processSolar } from "./utils/solar";
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

      this.LoadingText = "민감발언 탐지 중";
      let revisedData = await processSolar(scriptData);

      scriptData = scriptData.map(sentence => ({
        ...sentence,
        isModified: false
      }));

      let scriptIndex = 0;
      for (let sentence of revisedData) {
        while (scriptIndex < scriptData.length) {
          if (scriptData[scriptIndex].start === sentence.start) {
            scriptData[scriptIndex].isModified = true;
            break;
          }
          scriptIndex++;
        }

        sentence.thumbnail = await this.generateThumbnail(sentence.start);
      }

      this.transcript = scriptData;
      this.revised_transcript = revisedData;

      this.isLoading = false;
    },
    handleSentenceClick(startTime) {
      if (this.$refs.videoComponent) {
        this.$refs.videoComponent.seekToTime(startTime);
      }
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
    flex-direction: row;
    width: 100%;
    height: 100vh;
    padding: 20px;
    box-sizing: border-box;
    justify-content: space-between;
    align-items: center;
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
    margin: auto;
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

  .left-container {
    display: flex;
    flex-direction: column;
    width: 48%;
    align-items: center;
    gap: 15px;
  }

  .right-container {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    width: 48%;
    padding: 20px;
    box-sizing: border-box;
  }
</style>