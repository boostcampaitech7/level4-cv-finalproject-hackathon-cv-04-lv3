<template>
    <div class="revised-script">
      <h2>교정된 스크립트</h2>
      <div 
        v-for="(sentence, index) in transcript" 
        :key="index" 
        @click="handleClick(sentence)"
        class="sentence-block"
      >
        <div class="thumbnail">
          <img v-if="sentence.thumbnail" :src="sentence.thumbnail" alt="Video thumbnail" />
          <div class="time-info">{{ sentence.start }}s - {{ sentence.end }}s</div>
        </div>
        <div class="text-container">
          <div class="before">
            <span class="label">Before:</span> "{{ sentence.origin_text }}"
          </div>
          <div class="after">
            <span class="label">After:</span> "{{ sentence.new_text }}"
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    props: {
      transcript: {
        type: Array,
        default: () => [],
      },
    },
    methods: {
      handleClick(sentence) {
        console.log(`클릭한 문장의 시작 시간: ${sentence.start}초`);
        this.$emit("sentence-clicked", sentence.start);
      },
    },
  };
  </script>
  
  <style scoped>
  .revised-script {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    height: 100%;
    padding: 20px;
    box-sizing: border-box;
  }
  .sentence-block {
    display: flex;
    align-items: center;
    background-color: #f3f4f6;
    padding: 10px;
    margin: 10px 0;
    width: 100%;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.2s;
    text-align: left;
    font-size: 16px;
    border-left: 5px solid #007bff;
  }
  .sentence-block:hover {
    background-color: #e0e7ff;
  }
  .thumbnail {
    width: 120px;
    text-align: center;
    margin-right: 10px;
  }
  .thumbnail img {
    width: 100%;
    height: auto;
    border-radius: 5px;
  }
  .time-info {
    font-size: 12px;
    color: #555;
    margin-top: 5px;
  }
  .text-container {
    flex: 1;
  }
  .before, .after {
    padding: 5px;
    font-size: 16px;
  }
  .label {
    font-weight: bold;
    color: #555;
  }
  .before {
    color: #d9534f; /* 빨간색 */
  }
  .after {
    color: #5cb85c; /* 초록색 */
  }
  </style>
  