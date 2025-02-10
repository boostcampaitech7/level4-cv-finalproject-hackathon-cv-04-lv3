import { STT_API_URL, RAG_API_URL, TRANSFER_URL } from "./apiConfig";


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

// export async function processReward(revisedData) {
//     try {
//         const response = await fetch(REWARD_API_URL, {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({
//                 query: scriptData,
//                 k: 4,
//                 max_token: 3000,
//                 temperature: 0.0,
//                 chain_type: "stuff"
//             })
//         });

//         if (!response.ok) {
//             throw new Error(`요청 실패: ${response.status}`);
//         }

//         return await response.json();
//     } catch (error) {
//         console.error("RAG 요청 중 오류 발생:", error);
//         return [];
//     }
// }

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