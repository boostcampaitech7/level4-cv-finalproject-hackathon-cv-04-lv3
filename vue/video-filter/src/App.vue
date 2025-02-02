<template>
  <div class="app">
    <LoadingOverlay v-if="isLoading" :LoadingText="LoadingText" />
    <div class="left-container">
      <input type="file" @change="handleFileUpload" accept="video/*" />
      <Video ref="videoComponent" :videoUrl="videoUrl" :key="videoUrl" />
      <button v-if="videoFile" @click="convertToText">변환</button>
      <Script :transcript="transcript" @sentence-clicked="handleSentenceClick" />
    </div>
    <div class="right-container">
      <RevisedScript :transcript="revised_transcript" @sentence-clicked="handleSentenceClick" />
    </div>
  </div>
</template>

<script>
import Video from "./components/Video.vue";
import Script from "./components/Script.vue";
import RevisedScript from "./components/RevisedScript.vue";
import { processSTT, processEmotion, processSolar } from "./utils/api";
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
    async convertToText() {
      if (!this.videoFile) return;
      this.isLoading = true;

      this.LoadingText = "STT 변환 중"
      let scriptData = await processSTT(this.videoFile);

      this.LoadingText = "Emotion 감지지 중"
      let EmotionData = await processEmotion(this.videoFile);

      this.LoadingText = "민감발언 탐지 중"
      let revisedData = await processSolar(scriptData);
      
      scriptData = scriptData.map((sentence, index) => ({
        ...sentence,
        isModified: EmotionData[index],
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
