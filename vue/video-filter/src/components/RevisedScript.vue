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
    /* div 부분 @click="handleClick(sentence)" 삭제*/
    /*handleClick(sentence) {
      console.log(`클릭한 문장의 시작 시간: ${sentence.start}초`);
      this.$emit("sentence-clicked", sentence.start);
    },*/
    prevPage() {
      if (this.currentPage > 1) {
        this.currentPage--;
      }
    },
    nextPage() {
      if (this.currentPage < this.totalPages) {
        this.currentPage++;
      }
    }
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
</style>
