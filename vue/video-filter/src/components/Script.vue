<template>
    <div class="video-script">
        <h2>STT 변환 스크립트</h2>

        <div class="pagination">
            <button @click="prevPage" :disabled="currentPage === 1" class="arrow-button">◀</button>
            <span>{{ currentPage }} / {{ totalPages }}</span>
            <button @click="nextPage" :disabled="currentPage === totalPages" class="arrow-button">▶</button>
        </div>

        <p>
            <span
            v-for="(sentence, index) in paginatedTranscript" :key="index"
            @click="handleClick(sentence)" class="sentence" :class="getModifiedClass(sentence.choice)" >
                {{ sentence.text }}
            </span>
        </p>
    </div>
</template>
  
<script>
    export default {
        name:"ScriptComponent",
        props: {
            transcript: {
                type: Array,
                default: () => [],
            },
        },
        data() {
            return {
                currentPage: 1,
                maxCharsPerPage: 400,
                paginatedSentences: [],
            };
        },
        computed: {
            paginatedTranscript() {
                return this.paginatedSentences[this.currentPage - 1] || [];
            },
            totalPages() {
                return this.paginatedSentences.length;
            }
        },
        watch: {
            transcript: {
                immediate: true,
                handler(newTranscript) {
                    this.paginateTranscript(newTranscript);
                }
            }
        },
        methods: {
            handleClick(sentence) {
                this.$emit("sentence-clicked", sentence.start);
            },
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
            paginateTranscript(transcript) {
                this.paginatedSentences = [];
                let currentPage = [];
                let currentLength = 0;

                transcript.forEach(sentence => {
                    const sentenceLength = sentence.text.length;

                    if (currentLength + sentenceLength <= this.maxCharsPerPage || currentLength === 0) {
                        currentPage.push(sentence);
                        currentLength += sentenceLength;
                    } else {
                        this.paginatedSentences.push(currentPage);
                        currentPage = [sentence];
                        currentLength = sentenceLength;
                    }
                });

                if (currentPage.length > 0) {
                    this.paginatedSentences.push(currentPage);
                }

                this.currentPage = 1;
            },
            getModifiedClass(choice) {
                if (choice === 'O') return "text-o";
                if (choice === 'X') return "text-x";
                return "text-default";
            }
        },
    };
</script>
  
<style scoped>
  .video-script {
    width: 100%;
    text-align: center;
    margin-top: 20px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #f9f9f9;
    margin: 20px auto;
  }
  p {
    font-size: 18px;
    line-height: 1.6;
    text-align: justify;
    padding: 10px;
  }
  .sentence {
    cursor: pointer;
    padding: 2px 5px;
    transition: background-color 0.2s;
  }
  .sentence:hover {
    background-color: #dbeafe;
  }

  .text-o {
    color: #5cb85c; /* 초록색 */
    font-weight: bold;
  }
  .text-x {
    color: #d9534f; /* 빨간색 */
    font-weight: bold;
  }
  .text-default {
    color: black; /* 검은색 */
  }
</style>