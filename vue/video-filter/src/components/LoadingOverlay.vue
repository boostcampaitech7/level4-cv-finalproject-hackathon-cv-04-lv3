<template>
    <div class="overlay">
      <div class="spinner"></div>
      <p class="loading-text">{{ LoadingText }}...</p>
    </div>
  </template>
  
  <script>
  export default {
    name: "LoadingOverlayComponent",
    props: {
        LoadingText: String,
    }
  };
  </script>
  
  <style scoped>
  /* 🔹 반투명 오버레이 */
  .overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5); /* 반투명 검은색 */
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000; /* 기존 화면 위에 표시 */
  }
  
  /* 🔹 로딩 스피너 */
  .spinner {
    width: 50px;
    height: 50px;
    border: 5px solid #f3f3f3;
    border-top: 5px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  /* 🔹 로딩 텍스트 */
  .loading-text {
    color: white;
    font-size: 18px;
    font-weight: bold;
    margin-top: 15px;
    text-align: center;
  }
  </style>
  