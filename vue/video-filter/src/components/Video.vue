<template>
    <div>
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
  
<style>
</style>
  