# -*- coding: utf-8 -*-

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re

# --- 應用程式設定與標題 ---
# st.set_page_config 會設定頁面的基本屬性，如標題和佈局
# layout="wide" 可以讓內容佔滿整個螢幕寬度，更適合顯示表格數據
st.set_page_config(page_title="Qforia 查詢模擬器", layout="wide")
st.title("🔍 Qforia: AI 應用查詢擴展模擬器")
st.markdown("一個模擬生成式 AI 搜尋服務（如 Google AI Overview）在收到使用者查詢後，如何在背景生成多個相關子查詢的工具。")

# --- 常數定義 ---
# 將固定的提示模板定義為常數，方便管理與修改
PROMPT_TEMPLATE_HEADER = """
您正在模擬一個先進的生成式 AI 搜尋系統（如 Google 的 AI 模式）的查詢擴展（Query Fan-Out）過程。
使用者的原始查詢是："{q}"
選擇的模式是："{mode}"

**您的第一個任務是：根據以下指示，決定要生成的查詢總數，並說明理由。**
{num_queries_instruction}

**在決定了數量和理由後，請精確生成該數量的、獨特的、合成的查詢。**
在生成的查詢集合中，必須盡可能包含以下每種查詢轉換類型（如果總數允許）：
1.  **重新表述 (Reformulations)**：用不同的方式問同一個問題。
2.  **相關查詢 (Related Queries)**：探索與主題相關但非核心的面向。
3.  **隱性查詢 (Implicit Queries)**：挖掘使用者沒有明說但可能想知道的背景資訊。
4.  **比較性查詢 (Comparative Queries)**：對比不同選項的優劣。
5.  **實體擴展 (Entity Expansions)**：深入探討查詢中提到的特定實體（人、事、物）。
6.  **個人化查詢 (Personalized Queries)**：模擬基於使用者常見需求的查詢（例如：初學者指南、預算考量等）。

每個獨立查詢的 'reasoning' 欄位，都應該解釋為什麼要生成這個特定的查詢、它的類型，以及它如何對應到使用者的整體意圖。
請勿生成依賴即時使用者歷史紀錄或地理位置的查詢。

**請嚴格按照以下格式，僅回傳一個有效的 JSON 物件：**
"""

JSON_STRUCTURE_EXAMPLE = """
{
  "generation_details": {
    "target_query_count": 12, // 這是範例數字，您需要根據分析自行決定實際數字。
    "reasoning_for_count": "由於使用者查詢的複雜度中等，我選擇生成略多於基本數量的查詢，以涵蓋關鍵面向，例如 X、Y 和 Z。" // 這是範例理由，請提供您自己的分析。
  },
  "expanded_queries": [
    {
      "query": "範例查詢 1...",
      "type": "reformulation",
      "user_intent": "範例意圖...",
      "reasoning": "生成此查詢的具體理由..."
    }
  ]
}
"""

# --- 側邊欄：使用者輸入與設定 ---
st.sidebar.header("⚙️ 設定")
# 讓使用者輸入自己的 Gemini API Key，type="password" 會將輸入顯示為星號，保護隱私
gemini_key = st.sidebar.text_input("請輸入您的 Gemini API Key", type="password")
# 提供預設查詢，引導使用者操作
user_query = st.sidebar.text_area("輸入您的查詢", "帶家人去北海道五天四夜，有哪些推薦的行程？預算有限。", height=120)
# 讓使用者選擇模式，影響生成查詢的數量和複雜度
mode = st.sidebar.radio("查詢模式", ["AI 概覽 (簡單)", "AI 模式 (複雜)"], help="簡單模式生成較少但核心的查詢；複雜模式則會生成更多、更深入的查詢。")

# --- Gemini 模型設定 ---
model = None
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # 使用一個支援複雜 JSON 輸出的通用模型
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
    except Exception as e:
        st.error(f"API Key 設定失敗，請檢查您的 Key 是否正確: {e}")
        st.stop()
else:
    # 如果沒有輸入 API Key，提示使用者並停止執行
    st.info("請在左側側邊欄輸入您的 Gemini API Key 以開始使用。")
    st.stop()

# --- 核心功能函式 ---

def get_query_fanout_prompt(q, mode):
    """根據使用者查詢和模式，動態生成完整的提示(Prompt)。"""
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI 概覽 (簡單)":
        num_queries_instruction = (
            f"首先，分析使用者查詢：「{q}」。基於其複雜度和「{mode}」模式，"
            f"**您必須決定一個最佳的查詢生成數量。** "
            f"這個數字**至少需要是 {min_queries_simple}**。 "
            f"對於一個直接的查詢，生成約 {min_queries_simple}-{min_queries_simple + 2} 個查詢可能就足夠了。 "
            f"如果查詢包含幾個不同面向或常見的後續問題，目標可以設定在 {min_queries_simple + 3}-{min_queries_simple + 5} 個查詢。"
            f"請簡要說明您選擇這個特定數字的理由。查詢本身應範圍明確且高度相關。"
        )
    else:  # AI 模式 (複雜)
        num_queries_instruction = (
            f"首先，分析使用者查詢：「{q}」。基於其複雜度和「{mode}」模式，"
            f"**您必須決定一個最佳的查詢生成數量。** "
            f"這個數字**至少需要是 {min_queries_complex}**。 "
            f"對於需要從多角度、子主題、比較或更深層含義進行探索的多面向查詢，"
            f"您應該生成更全面的集合，可能達到 {min_queries_complex + 5}-{min_queries_complex + 10} 個查詢，如果查詢特別廣泛或深入，甚至可以更多。"
            f"請簡要說明您選擇這個特定數字的理由。查詢應具備多樣性和深度。"
        )

    # 組合完整的提示
    full_prompt = PROMPT_TEMPLATE_HEADER.format(
        q=q,
        mode=mode,
        num_queries_instruction=num_queries_instruction
    ) + JSON_STRUCTURE_EXAMPLE
    
    return full_prompt

def generate_fanout(query, mode):
    """呼叫 Gemini API，生成並解析查詢擴展的結果。"""
    prompt = get_query_fanout_prompt(query, mode)
    
    try:
        # 呼叫 API
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # 使用正規表示式從回應中提取 JSON 內容
        # re.DOTALL 讓 '.' 可以匹配換行符，以應對跨行的 JSON
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if not json_match:
            st.error("🔴 模型回應中未找到有效的 JSON 物件。")
            st.text_area("模型原始回應", raw_text, height=200)
            return None

        json_text = json_match.group(0)
        data = json.loads(json_text)
        
        # 從解析後的資料中獲取所需資訊
        generation_details = data.get("generation_details", {})
        expanded_queries = data.get("expanded_queries", [])

        # 使用 session_state 來儲存生成細節，以便在主介面顯示
        st.session_state.generation_details = generation_details

        return expanded_queries

    except json.JSONDecodeError as e:
        st.error(f"🔴 解析 Gemini 回應的 JSON 時失敗: {e}")
        st.text_area("導致錯誤的原始 JSON 內容", json_text if 'json_text' in locals() else "N/A", height=200)
        st.session_state.generation_details = None
        return None
    except Exception as e:
        st.error(f"🔴 生成過程中發生未預期的錯誤: {e}")
        if 'raw_text' in locals():
            st.text_area("模型原始回應", raw_text, height=200)
        st.session_state.generation_details = None
        return None

# --- 主應用程式邏輯 ---

# 初始化 session_state，避免在第一次執行時出錯
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None
if 'results' not in st.session_state:
    st.session_state.results = None

# 當使用者點擊按鈕時，觸發生成流程
if st.sidebar.button("執行查詢擴展 🚀"):
    # 清除舊的結果
    st.session_state.generation_details = None
    st.session_state.results = None
    
    if not user_query.strip():
        st.warning("⚠️ 請先輸入一個查詢。")
    else:
        # 顯示載入動畫，提升使用者體驗
        with st.spinner("🤖 正在呼叫 Gemini 生成查詢擴展... 請稍候..."):
            st.session_state.results = generate_fanout(user_query, mode)

# --- 結果顯示 ---

# 只有在 session_state 中有結果時才顯示
if st.session_state.results:
    results = st.session_state.results
    st.success("✅ 查詢擴展完成！")

    # 顯示模型的生成計畫
    if st.session_state.generation_details:
        details = st.session_state.generation_details
        generated_count = len(results)
        target_count_model = details.get('target_query_count', 'N/A')
        reasoning_model = details.get('reasoning_for_count', '模型未提供')

        with st.expander("🧠 查看模型的生成計畫", expanded=True):
            st.markdown(f"🔹 **模型目標數量：** `{target_count_model}`")
            st.markdown(f"🔹 **模型決策理由：** *{reasoning_model}*")
            st.markdown(f"🔹 **實際生成數量：** `{generated_count}`")
            
            # 如果目標與實際不符，顯示警告
            if isinstance(target_count_model, int) and target_count_model != generated_count:
                st.warning(f"⚠️ 注意：模型計畫生成 {target_count_model} 個查詢，但實際產生了 {generated_count} 個。")
    
    st.markdown("---")
    st.subheader("📊 生成的查詢結果")

    # 將結果轉換為 DataFrame 以便於顯示
    df = pd.DataFrame(results)
    # 重新排列欄位順序，讓呈現更直觀
    if not df.empty:
        df = df[['query', 'type', 'user_intent', 'reasoning']]
    
    # 使用 st.dataframe 顯示表格，並設定動態高度
    st.dataframe(df, use_container_width=True, height=(min(len(df), 20) + 1) * 35 + 3)

    # 提供 CSV 下載功能
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 下載 CSV 檔案",
        data=csv,
        file_name="qforia_output.csv",
        mime="text/csv",
    )

elif st.session_state.results is not None: # 處理 generate_fanout 回傳空列表的情況
    st.warning("⚠️ 模型沒有生成任何查詢，或回傳了一個空列表。")

# 如果沒有任何結果（初始狀態），顯示歡迎訊息
if st.session_state.results is None and st.session_state.generation_details is None:
    st.markdown("---")
    st.info(
        """
        **歡迎使用 Qforia！**

        1.  在左側的側邊欄輸入您的 **Gemini API Key**。
        2.  在 **"輸入您的查詢"** 欄位中寫下您想分析的任何問題。
        3.  選擇您想要的 **查詢模式**。
        4.  點擊 **"執行查詢擴展"** 按鈕，即可在此處看到結果。
        """
    )

# --- 頁尾 ---
st.markdown("---")
st.markdown("Made with ❤️ by a Streamlit enthusiast.")

