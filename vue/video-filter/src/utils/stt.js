export async function processSTT(videoFile) {
    console.log(videoFile);

    return new Promise((resolve) => {
        setTimeout(() => {
            resolve([
                { start: 0, end: 3, text: "안녕하세요." },
                { start: 4, end: 7, text: "이것은 STT 테스트입니다." },
                { start: 8, end: 12, text: "비디오 파일을 분석 중입니다." },
                { start: 13, end: 16, text: "결과를 확인해주세요." },
                { start: 16, end: 19, text: "2결과를 확인해주세요." },
                { start: 19, end: 21, text: "3결과를 확인해주세요." },
            ]);
        }, 2000);
    });
}