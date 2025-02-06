import { EMOTION_API_URL } from "./apiConfig";//STT_API_URL//
/*
export async function processSTT(videoFile) {
    if (!videoFile) {
        console.error("비디오가 존재하지 않습니다.");
        return;
    }

    const formData = new FormData();
    formData.append("file", videoFile);

    try {
        const response = await fetch(STT_API_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`STT 요청 실패: ${response.status}`);
        }

        const result = await response.json();

        return result;
    }catch (error) {
        console.error("STT 요청 중 오류 발생:", error);
    }
}
*/

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
         }, 1000);
     });
 }

export async function processEmotion(videoFile) {
    if (!videoFile) {
        console.error("비디오가 존재하지 않습니다.");
        return;
    }

    const formData = new FormData();
    formData.append("file", videoFile);

    try {
        const response = await fetch(EMOTION_API_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Emotion 요청 실패: ${response.status}`);
        }

        const result = await response.json();

        return result;
    }catch (error) {
        console.error("Emotion 요청 중 오류 발생:", error);
    }
}

export async function processSolar(transcript) {
    console.log(transcript);

    return new Promise((resolve) => {
        setTimeout(() => {
            resolve([
                { start: 0, end: 3, origin_text:"안녕하세요.", new_text: "반가워요요." ,reason: "비속어 또는 논란 언어가 포함되어 있습니다."},
                { start: 4, end: 8, origin_text: "이것은 STT 테스트입니다.", new_text: "텍스트트 파일을 분석 중입니다." , reason: "엄처어어어어어어어어어엉 긴텍스트트트트트트트트트으으으으으으으ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ"},
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
                { start: 13, end: 16, origin_text: "결과를 확인해주세요.", new_text: "변화가 있나요?" },
            ]);
        }, 1000);
    });
}
