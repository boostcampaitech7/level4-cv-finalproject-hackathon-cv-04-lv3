<template>
    <div class="video-container">
        <video v-if="videoUrl" ref="videoPlayer" controls>
            <source :src="videoUrl" type="video/mp4" />
            브라우저가 비디오 태그를 지원하지 않습니다.
        </video>
    </div>
</template>
  
<script>
    export default {
        name: "VideoComponent",
        props: {
            videoUrl: String,
        },
        methods: {
            seekToTime(time) {
                if (this.$refs.videoPlayer) {
                    this.$refs.videoPlayer.currentTime = time;
                    this.$refs.videoPlayer.play();
                }
            }
        },
    };
</script>
  
<style scoped>
    .video-container {
        max-width: 100%;
        max-height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
    }

    video {
        max-width: 100%; /* 부모 컨테이너 크기에 맞춤 */
        max-height: 100%; /* 비율 유지 */
        object-fit: contain;
        border-radius: 8px; /* 부드러운 모서리 */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
</style>
  