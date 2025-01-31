<template>
    <div class="video-script">
        <h2>STT 변환 스크립트</h2>
        <p>
            <span
            v-for="(sentence, index) in transcript" :key="index"
            @click="handleClick(sentence)" class="sentence" :class="{ modified: sentence.isModified }" >
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
        methods: {
            handleClick(sentence) {
                this.$emit("sentence-clicked", sentence.start);
            },
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

  .modified {
    color: red;
    font-weight: bold;
    }
</style>
  