<template>
  <div class="revised-script">
    <h2>교정된 스크립트</h2>

    <div class="pagination">
      <button class="arrow-button" @click="prevPage" :disabled="currentPage === 1">◀</button>
      <span>{{ currentPage }} / {{ totalPages }}</span>
      <button class="arrow-button" @click="nextPage" :disabled="currentPage === totalPages">▶</button>
    </div>

    <div 
      v-for="(sentence, index) in paginatedTranscript" 
      :key="index" 
      class="sentence-block"
    >
      <div class="thumbnail">
        <img v-if="sentence.thumbnail" :src="sentence.thumbnail" alt="Video thumbnail" />
        <div class="time-info">{{ sentence.start }}s - {{ sentence.end }}s</div>
        <div class="buttons">
          <img 
            src="@/assets/off-o.png" 
            @click="sentence.choice !== 'O' && updateScript(sentence, 'O')" 
            alt="O 버튼" 
          />
          <img 
            src="@/assets/off-x.png" 
            @click="sentence.choice !== 'X' && updateScript(sentence, 'X')" 
            alt="X 버튼" 
          />
        </div>
      </div>
      <div class="text-container">
        <div class="before">
          <span class="label">Before:</span> "{{ sentence.origin_text }}"
        </div>
        <div class="after">
          <span class="label">After:</span> "{{ sentence.new_text }}"
        </div>
        <div class="reason" v-if="sentence.reason">
          <img src="@/assets/solar.png" alt="Reason 이미지" />
          "{{ sentence.reason }}"
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
  data() {
    return {
      currentPage: 1,
      pageSize: 4,
    }
  },
  computed: {
    paginatedTranscript() {
      const start = (this.currentPage - 1) * this.pageSize;
      return this.transcript.slice(start, start + this.pageSize);
    },
    totalPages() {
      return Math.ceil(this.transcript.length / this.pageSize);
    }
  },
  methods: {
    prevPage() {
      if (this.currentPage > 1) {
        this.currentPage--;
      }
    },
    nextPage() {
      if (this.currentPage < this.totalPages) {
        this.currentPage++;
      }
    },
    updateScript(sentence, choice) {
      this.$emit('update-script', { sentence, choice });
    }
  }
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
  /*cursor: pointer;*/
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
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: 15px;
}
.arrow-button {
  padding: 8px 12px;
  font-size: 18px;
  margin: 0 10px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.2s;
}
.arrow-button:hover {
  background-color: #0056b3;
  transform: scale(1.1);
}
.arrow-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  transform: scale(1);
}
.pagination span {
  font-size: 16px;
  font-weight: bold;
  margin: 0 10px;
}

.buttons {
  display: flex;
  gap: 0px; /* 버튼 사이 간격을 조정 */
  margin-top: 10px;
}
.buttons img {
  width: 50px; /* 원하는 가로 길이로 설정 */
  height: 30px; /* 원하는 세로 길이로 설정 */
  cursor: pointer;
  border: 2px solid transparent; /* 기본 테두리 색상 */
}
.buttons img.active {
  border-color: red; /* 활성화된 버튼의 테두리 색상 */
}
.text-o {
  color: #5cb85c; /* 초록색 */
}
.text-x {
  color: #d9534f; /* 빨간색 */
}

.text-default {
  color: black; /* 검은색 */
}

.text-container {
  flex: 1;
}

.reason {
  margin-top: 10px;
  padding: 10px;
  background-color: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
  border-radius: 5px;
}

.reason img {
  width: 24px; /* 원하는 가로 길이로 설정 */
  height: 24px; /* 비율에 맞게 높이 자동 설정 */
  vertical-align: middle; /* 텍스트와 이미지 수직 정렬 */
  margin-right: 5px; /* 이미지와 텍스트 사이 간격 */
}

</style>
