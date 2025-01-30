export async function processSolar(transcript) {
    console.log(transcript);

    return new Promise((resolve) => {
        setTimeout(() => {
            resolve([
                { start: 0, end: 3, origin_text:"안녕하세요.", new_text: "반가워요요." },
                { start: 8, end: 12, origin_text: "이것은 STT 테스트입니다.", new_text: "텍스트트 파일을 분석 중입니다." },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" }
            ]);
        }, 2000);
    });
}