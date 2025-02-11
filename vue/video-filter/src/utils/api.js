import { STT_API_URL, REWARD_API_URL, SENTIMENT_URL, RAG_API_URL, TRANSFER_URL } from "./apiConfig";


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


// export async function processEmotion(videoFile) {
//     if (!videoFile) {
//         console.error("비디오가 존재하지 않습니다.");
//         return;
//     }

//     const formData = new FormData();
//     formData.append("file", videoFile);

//     try {
//         const response = await fetch(EMOTION_API_URL, {
//             method: "POST",
//             body: formData,
//         });

//         if (!response.ok) {
//             throw new Error(`Emotion 요청 실패: ${response.status}`);
//         }

//         const result = await response.json();

//         return result;
//     }catch (error) {
//         console.error("Emotion 요청 중 오류 발생:", error);
//     }
// }

export async function processSolar(scriptData) {
    try {
        const response = await fetch(`${RAG_API_URL}/rag/similarity`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: scriptData,
                k: 4,
                max_token: 3000,
                temperature: 0.0,
                chain_type: "stuff"
            })
        });

        if (!response.ok) {
            throw new Error(`요청 실패: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error("RAG 요청 중 오류 발생:", error);
        return [];
    }
}

export async function processReward(revisedData) {

    try {
        // const titleList = revisedData.map(item => item.title ?? "");
        const titleList = revisedData.map(() => "");
        const textList = revisedData.map(item => item.origin_text ?? "");
        const newList = revisedData.map(item => item.new_text ?? "");

        const response_origin = await fetch(REWARD_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title_list: titleList,
                text_list: textList
            })
        });

        const response_new = await fetch(REWARD_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title_list: titleList,
                text_list: newList
            })
        });

        const response_origin_2 = await fetch(SENTIMENT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title_list: titleList,
                text_list: textList
            })
        });

        const response_new_2 = await fetch(SENTIMENT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title_list: titleList,
                text_list: newList
            })
        });

        if (!response_origin.ok || !response_new.ok) {
            throw new Error(`요청 실패: ${response_origin.status}`);
        }

        // const [originData1, newData1] = await Promise.all([
        //     response_origin.json(),
        //     response_new.json(),
        // ]);

        // return [originData1, newData1];

        const [originData1, newData1, originData2, newData2] = await Promise.all([
            response_origin.json(),
            response_new.json(),
            response_origin_2.json(),
            response_new_2.json(),
        ]);
        
        // originData1이 2D 배열이므로, 각 문장별로 originData2의 값을 추가
        const combinedOriginData = originData1.map((values, index) => {
            const extraValue = originData2.results[index];  // originData2가 부족하면 0으로 채움
            return [...values, extraValue];  // 기존 3개 값 + originData2 값 추가
        });

        // newData는 구조가 동일하다고 가정하고 동일하게 처리
        const combinedNewData = newData1.map((values, index) => {
            const extraValue = newData2.results[index];
            return [...values, extraValue];
        });
        
        return [combinedOriginData, combinedNewData];

        // return await [response_origin.json(), response_new.json()];
    } catch (error) {
        console.error("BERT 요청 중 오류 발생:", error);
        return [];
    }
}

export async function sound_transfer(videoFile, changedScripts) {
    if (!videoFile) {
        console.error("비디오가 존재하지 않습니다.");
        return;
    }

    const formData = new FormData();
    formData.append("file", videoFile);
    formData.append("changed_scripts", JSON.stringify(changedScripts));

    try {
        const response = await fetch(TRANSFER_URL + "/sound_transfer/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Sound_Transfer 요청 실패: ${response.status}`);
        }

        const videoBlob = await response.blob();
        return URL.createObjectURL(videoBlob);
    } catch (error) {
        console.error("Sound_Transfer 요청 중 오류 발생:", error);
    }
}