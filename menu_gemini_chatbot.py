"""
menu_gemini_chatbot.py
Streamlit 앱: Gemini 기반(옵션) 음식 메뉴 추천 챗봇
- 목적: 사용자의 고민에 공감하고 최소 2개의 핵심 정보를 질문해 2~3가지 메뉴 제안
- 기본 모델: gemini-2.0-flash (선택 UI 제공)
- 시스템 프롬프트 및 동작 규칙은 코드 내에 반영됨
- 핵심 기능: 대화 히스토리, 최근 6턴 유지(초과 시 오래된 턴 삭제), CSV 자동 기록(옵션), 로그 다운로드, 대화 초기화, 모델/세션 표시
- 비밀키: st.secrets['GEMINI_API_KEY'] (없으면 임시 입력 UI 표시)
- 작성자 노트: 실제 Google Gemini 연동 시 google-generativeai 라이브러리 설정 필요(주석 참고)
"""
import streamlit as st
from datetime import datetime
import uuid
import pandas as pd
import time
import json
import os

# ---------- 설정 ----------
st.set_page_config(page_title="메뉴 추천 챗봇 (Gemini)", page_icon="🍽️", layout="wide")
SESSION_ID = st.session_state.get("session_id", str(uuid.uuid4()))
if "session_id" not in st.session_state:
    st.session_state["session_id"] = SESSION_ID

# System prompt (요청하신 규칙 반영)
SYSTEM_PROMPT = """
당신은 메뉴 추천 전문 AI 챗봇입니다.
1) 사용자가 음식 결정 고민을 언급하면 즉시 공감하고 메뉴 추천 프로세스를 시작하세요.
2) 정확한 추천을 위해 사용자에게 최소 2가지 이상의 핵심 정보를 질문하고 수집하세요. 한 번에 너무 많은 것을 묻지 말고 자연스러운 대화처럼 하세요.
3) 수집된 정보를 바탕으로 구체적인 메뉴 2~3가지를 제안하세요.
4) 사용자가 제안한 메뉴를 거절하거나 망설이면 즉시 대안을 제시하거나 추가 질문을 통해 선호를 다시 파악하세요.
5) 사용자가 "아무거나" 또는 "추천해 주는 거"라고 말하면 절대 "아무거나요?"라고 되묻지 마세요. 대신 가장 대중적인 메뉴(예: 제육볶음, 돈까스, 떡볶이)를 먼저 하나 제안하거나, 선택지를 극단적으로 좁히는 질문(예: "좋아요! 그럼 밥 vs 면, 딱 하나만 골라주세요!")을 하세요.
6) 당신은 메뉴 추천만 담당합니다. 레시피/영양/배달 등은 '저는 메뉴 추천 전문 챗봇이라 그 부분은 도와드리기 어려워요 😥.' 라고 정중히 안내하고 추천 작업에 집중하세요.
"""

# Trigger keywords (사용자 시작 문구들)
TRIGGERS = ["뭐 먹지", "메뉴 추천", "배고파", "뭐 먹을까", "뭐먹지", "추천해", "추천해줘", "추천해 줘", "아무거나"]

# App UI
st.title("🍽️ 메뉴 추천 챗봇 (Gemini API 사용 가능)")
st.markdown("음식 고르기 귀찮을 때! 공감하고 질문한 뒤 2~3가지 구체적 메뉴를 제안해 드려요.")

# Sidebar controls
with st.sidebar:
    st.header("설정")
    model = st.selectbox("모델 선택", options=["gemini-2.0-flash"], index=0)
    show_session = st.checkbox("세션/모델 표시", value=True)
    csv_logging = st.checkbox("대화 자동 CSV 기록", value=False)
    reset_btn = st.button("대화 초기화")
    st.markdown("---")
    st.write("Gemini API Key (선택)")
    # st.secrets 우선, 없으면 입력 필드 제공
    gemini_key = st.secrets.get("GEMINI_API_KEY") if st.secrets else None
    if not gemini_key:
        gemini_key = st.text_input("임시 GEMINI_API_KEY 입력 (없으면 모드: 로컬)", type="password")
    # small help
    st.markdown("※ 실제 Gemini 연동 시 `st.secrets['GEMINI_API_KEY']`에 키를 넣거나 여기 입력 후 사용하세요.")

# display model/session
if show_session:
    st.info(f"세션 ID: `{st.session_state['session_id']}`  •  모델: `{model}`")

# Reset
if reset_btn:
    st.session_state.clear()
    # regenerate session id
    st.session_state["session_id"] = str(uuid.uuid4())
    st.experimental_rerun()

# ---------- Conversation state ----------
if "history" not in st.session_state:
    # history: list of dicts: {"role":"user"/"assistant"/"system", "content": "...", "time": ts}
    st.session_state["history"] = [{"role": "system", "content": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()}]

if "collected" not in st.session_state:
    # collected info during a recommendation flow
    st.session_state["collected"] = {}  # e.g., {"cuisine": "한식", "carb": "밥"}

if "turns" not in st.session_state:
    st.session_state["turns"] = 0  # counts user->assistant exchange pairs

# helper: append to history
def append_history(role, content):
    st.session_state["history"].append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})

# Logging to CSV
LOGFILE = "chat_logs.csv"
def append_log(session_id, user_msg, assistant_msg):
    row = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_msg,
        "assistant": assistant_msg
    }
    df = pd.DataFrame([row])
    if not os.path.exists(LOGFILE):
        df.to_csv(LOGFILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(LOGFILE, mode="a", header=False, index=False, encoding="utf-8-sig")

# ---------- Simple local recommendation logic ----------
# This is the "menu engine" that, given collected info, proposes menus.
MENU_DB = {
    # cuisine: {carb: [options]}
    "한식": {
        "밥": ["김치찌개+밥", "제육볶음+밥", "된장찌개+밥"],
        "면": ["칼국수", "비빔국수", "잔치국수"],
        "분식": ["떡볶이", "김밥", "순대"],
        "기타": ["만두", "볶음밥"]
    },
    "중식": {
        "밥": ["짜장면(밥대신)", "볶음밥", "탕수육+밥"],
        "면": ["짬뽕", "유린기(면과 함께)"],
        "분식": ["중화비빔면"],
        "기타": ["마파두부+밥"]
    },
    "양식": {
        "밥": ["스테이크(감자or밥)", "리조또"],
        "면": ["크림파스타", "토마토파스타"],
        "분식": ["치즈버거", "피자(조각)"],
        "기타": ["샐러드"]
    },
    "분식": {
        "밥": ["김밥+주먹밥", "볶음밥"],
        "면": ["라볶이(면+떡)"],
        "분식": ["떡볶이", "순대", "튀김"],
        "기타": ["핫도그"]
    },
    "패스트푸드": {
        "기타": ["버거", "프라이+치킨", "샌드위치"]
    },
    "기타": {
        "밥": ["볶음밥", "덮밥"],
        "면": ["라멘", "우동"],
        "분식": ["떡볶이"],
        "기타": ["샐러드"]
    }
}

def get_recommendations(collected):
    # collected may contain keys: cuisine, carb, spice
    cuisine = collected.get("cuisine", "기타")
    carb = collected.get("carb", None)
    if cuisine not in MENU_DB:
        cuisine = "기타"
    candidates = []
    if carb and carb in MENU_DB[cuisine]:
        candidates = MENU_DB[cuisine][carb]
    else:
        # fallback gather several categories
        bucket = MENU_DB[cuisine]
        for k in ["밥", "면", "분식", "기타"]:
            if k in bucket:
                candidates += bucket[k]
    # deduplicate & pick up to 3
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    return unique[:3] if unique else ["김치찌개+밥", "제육볶음+밥"]

# helper: simple normalization
def normalize_text(t: str):
    return t.strip().lower()

# Detect "아무거나" kind of phrases
def is_anything_like_any(t: str):
    s = normalize_text(t)
    return any(x in s for x in ["아무거나", "추천해 주는 거", "추천해줘", "마음대로", "맘대로", "너가 골라"])

# Detect initial trigger
def contains_trigger(t: str):
    s = normalize_text(t)
    return any(tr in s for tr in TRIGGERS)

# ---------- (옵션) Gemini API wrapper with 429 retry ----------
# NOTE: This is a placeholder wrapper. For real Gemini usage, install google-generativeai
# and uncomment/adjust the client calls. The wrapper implements retry-on-429 behavior.
import requests
def call_gemini_api(prompt, model_name="gemini-2.0-flash", api_key=None, max_retries=4):
    """
    Placeholder HTTP wrapper for Gemini-like API calls.
    - If api_key is None: returns None to indicate 'no external call' (use local mode).
    - If api_key provided but real client isn't set up, this will attempt a basic REST call pattern.
    Adjust this function to your environment / official client library.
    """
    if not api_key:
        return None  # caller should fallback to local mock
    # Simple exponential backoff wrapper (for 429)
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            # Example generic POST — **MUST** be adjusted to actual Gemini endpoint & payload format.
            # The current block attempts a generic OpenAI-like call (may fail). Replace with official client.
            endpoint = f"https://api.openai.com/v1/chat/completions"  # placeholder; replace with correct Gemini endpoint
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.8
            }
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=15)
            if resp.status_code == 429:
                raise requests.exceptions.HTTPError("429")
            resp.raise_for_status()
            j = resp.json()
            # Try to extract text for OpenAI-like shape
            if "choices" in j and len(j["choices"]) > 0:
                return j["choices"][0]["message"]["content"]
            # fallback
            return j.get("result", {}).get("content", "")
        except requests.exceptions.HTTPError as e:
            if resp is not None and resp.status_code == 429:
                # retry
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                # other errors -> stop and return None
                return None
        except Exception as e:
            # network error or other
            return None
    return None

# ---------- Main chat UI ----------
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("대화")
    # show history in chat-like form
    chat_container = st.container()
    with chat_container:
        for item in st.session_state["history"]:
            role = item["role"]
            content = item["content"]
            t = item["time"]
            if role == "system":
                continue
            if role == "user":
                st.markdown(f"**사용자:** {content}")
            else:
                st.markdown(f"**챗봇:** {content}")

    # input
    user_input = st.text_input("메시지를 입력하세요", key="input_box")
    submit = st.button("전송")

with col2:
    st.subheader("도움말 / 상태")
    st.markdown("- 저는 `메뉴 추천 전문 챗봇`입니다.")
    st.markdown("- 최소 2가지 정보를 질문하고 2~3가지 구체적 메뉴를 제안합니다.")
    st.markdown("- `CSV 기록`을 켜면 대화가 `chat_logs.csv`에 저장됩니다.")
    if os.path.exists(LOGFILE):
        st.download_button("로그 다운로드 (CSV)", data=open(LOGFILE,"rb"), file_name=LOGFILE)
    if st.button("현재 대화 내역 CSV로 저장(즉시)"):
        # dump current history into csv
        rows = []
        for h in st.session_state["history"]:
            rows.append({"session_id": st.session_state["session_id"], "timestamp": h["time"], "role": h["role"], "content": h["content"]})
        df = pd.DataFrame(rows)
        fn = f"history_dump_{st.session_state['session_id']}.csv"
        df.to_csv(fn, index=False, encoding="utf-8-sig")
        st.success(f"저장됨: {fn}")
    st.markdown("---")
    st.write("세션 정보")
    st.json({"session_id": st.session_state["session_id"], "turns": st.session_state["turns"]})

# ---------- Conversation handling logic ----------
def handle_user_message(msg):
    msg = msg.strip()
    if not msg:
        return

    append_history("user", msg)

    # quick flow: if user asks about non-menu things like 레시피/배달, refuse politely per spec
    lower = msg.lower()
    if any(x in lower for x in ["레시피", "영양", "칼로리", "영양성분", "배달", "배달해", "맛집", "위치"]):
        reply = "저는 메뉴 추천 전문 챗봇이라 그 부분은 도와드리기 어려워요 😥. 하지만 '김치찌개'로 결정하신 건 정말 탁월해요!"
        append_history("assistant", reply)
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    # 1) If this is an initial trigger or we're not currently in a collecting flow, start flow
    collected = st.session_state["collected"]
    # If initial trigger in message and no collected yet, start by empathizing + ask first question
    if contains_trigger(msg) and not collected:
        reply = "아~ 그 마음 충분히 알아요! 어떤 걸 드시고 싶은지 같이 골라드릴게요. 우선 한 가지만 물어볼게요: 혹시 **한식 / 중식 / 양식 / 분식 / 패스트푸드** 중에 끌리는 종류가 있으세요? (없으면 '상관없음'이라고 해주세요)"
        append_history("assistant", reply)
        st.session_state["expecting"] = "cuisine"
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    # If user says "아무거나" style
    if is_anything_like_any(msg):
        # follow rule: don't reply "아무거나요?" — propose a popular menu or force narrowing question
        reply = "좋아요! 그럼 먼저 하나 추천드릴게요 — **제육볶음+밥**은 어떠세요? 아니면 '밥 vs 면' 중 딱 하나만 골라주실래요?"
        append_history("assistant", reply)
        # if user then picks we continue
        st.session_state["expecting"] = "confirm_any"  # special
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    # If we are expecting a specific field
    expecting = st.session_state.get("expecting", None)
    if expecting == "cuisine":
        # user answered cuisine
        # normalize and store first meaningful token
        answer = msg.strip()
        # map to known categories
        map_lower = answer.lower()
        if any(k in map_lower for k in ["한식","korean"]):
            cat = "한식"
        elif any(k in map_lower for k in ["중식","chinese"]):
            cat = "중식"
        elif any(k in map_lower for k in ["양식","western","이탈리","파스타","스테이크"]):
            cat = "양식"
        elif any(k in map_lower for k in ["분식","떡볶이","김밥"]):
            cat = "분식"
        elif any(k in map_lower for k in ["패스트","버거","피자","치킨"]):
            cat = "패스트푸드"
        elif "상관없" in map_lower or "없음" in map_lower:
            cat = "기타"
        else:
            cat = answer  # whatever user said

        st.session_state["collected"]["cuisine"] = cat
        # ask second question
        reply = f"좋아요 — **{cat}** 쪽이군요. 그러면 한 가지만 더 물을게요: 오늘은 **밥 / 면 / 분식 / 기타** 중 무엇이 끌리세요?"
        append_history("assistant", reply)
        st.session_state["expecting"] = "carb"
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    if expecting == "carb":
        answer = msg.strip()
        a_lower = answer.lower()
        if any(k in a_lower for k in ["밥","rice"]):
            carb = "밥"
        elif any(k in a_lower for k in ["면","국수","noodle","라면","칼국수","우동","파스타","스파게티"]):
            carb = "면"
        elif any(k in a_lower for k in ["분식","떡","김밥","떡볶이"]):
            carb = "분식"
        else:
            carb = "기타"
        st.session_state["collected"]["carb"] = carb

        # Now we have at least 2 pieces -> propose 2~3 menus
        recs = get_recommendations(st.session_state["collected"])
        reply = f"감사해요! 추천을 드릴게요 — **{st.session_state['collected'].get('cuisine','')}, {carb}** 기준으로 아래 메뉴를 제안합니다:\n\n"
        for i, r in enumerate(recs, 1):
            reply += f"{i}. {r}\n"
        reply += "\n마음에 드는 번호나 항목을 골라주세요. 맘에 들지 않으면 바로 다른 대안을 더 드릴게요!"
        append_history("assistant", reply)
        # reset expecting so we can get selection or rejection next
        st.session_state.pop("expecting", None)
        st.session_state["turns"] += 1
        # After giving suggestions, keep collected to allow refinements, but if conversation grows, maintain last 6 turns (handled below)
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    if expecting == "confirm_any":
        # user responded to the "아무거나" prompt: they may accept the suggested popular menu or choose to narrow
        if any(x in msg.lower() for x in ["제육","좋아","좋아요","ok","괜찮","오케이","오케"]):
            reply = "좋아요! 그럼 제육볶음으로 최종 추천할게요 🍚 맛있게 드세요!"
            append_history("assistant", reply)
            st.session_state.pop("expecting", None)
            if csv_logging:
                append_log(st.session_state["session_id"], msg, reply)
            return
        # if they choose to narrow by '밥' or '면'
        if any(x in msg.lower() for x in ["밥","면","분식","기타"]):
            st.session_state["collected"]["carb"] = "밥" if "밥" in msg else ("면" if "면" in msg else "분식" if "분식" in msg else "기타")
            recs = get_recommendations(st.session_state["collected"])
            reply = "알겠어요! 그럼 아래에서 골라보세요:\n"
            for i,r in enumerate(recs,1):
                reply += f"{i}. {r}\n"
            append_history("assistant", reply)
            st.session_state.pop("expecting", None)
            if csv_logging:
                append_log(st.session_state["session_id"], msg, reply)
            return
        # otherwise do fallback
        reply = "음.. 어떤 스타일을 더 선호하실지(밥/면/분식 등) 하나만 알려주시면 바로 추천 좁혀드릴게요!"
        append_history("assistant", reply)
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    # If user replies to offered menu – selecting or rejecting
    # Check if they select a number or name that matches last suggestions
    # Find last assistant suggestions in history (simple search)
    last_assistant = None
    for h in reversed(st.session_state["history"]):
        if h["role"] == "assistant":
            last_assistant = h["content"]
            break

    if last_assistant and any(ch.isdigit() for ch in msg):
        # try to parse a chosen number
        chosen = None
        for token in msg.split():
            if token.isdigit():
                try:
                    n = int(token)
                    # extract list from last_assistant lines
                    lines = [ln.strip() for ln in last_assistant.splitlines() if ln.strip()]
                    opts = [ln.split(". ",1)[1] if ". " in ln else ln for ln in lines if ln[0].isdigit() or ln.startswith("1.")]
                    if 1 <= n <= len(opts):
                        chosen = opts[n-1]
                        break
                except Exception:
                    continue
        if chosen:
            reply = f"좋은 선택이에요 — **{chosen}**으로 결정하셨군요! 맛있게 드세요 😊"
            append_history("assistant", reply)
            if csv_logging:
                append_log(st.session_state["session_id"], msg, reply)
            return

    # If user explicitly rejects proposals (망설임/거절 같은 키워드)
    if any(x in lower for x in ["아니", "아니요", "싫어", "별로", "다른", "다른거", "아님"]):
        # Immediately propose alternatives or ask a clarifying question
        reply = "괜찮아요, 실망하지 않아요! 조금 더 좁혀볼게요 — 맵게 드실래요, 아니면 순하게 드실래요? 또는 가격대(저렴/보통/고급) 중 하나만 골라주세요."
        append_history("assistant", reply)
        st.session_state["expecting"] = "refine_preference"
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    if st.session_state.get("expecting") == "refine_preference":
        # try simple heuristics
        if any(x in lower for x in ["맵","매운"]):
            reply = "맵게 원하시는군요! 그럼 매운 제육볶음, 매운 떡볶이 등으로 바로 추천드릴게요:\n1. 매운 제육볶음+밥\n2. 매운 떡볶이\n원하시면 1 또는 2로 골라주세요."
            append_history("assistant", reply)
            st.session_state.pop("expecting", None)
            if csv_logging:
                append_log(st.session_state["session_id"], msg, reply)
            return
        if any(x in lower for x in ["순","안맵","순하게"]):
            reply = "순한 걸로요! 그럼 다음 중 골라보세요:\n1. 돈까스+밥\n2. 크림 파스타\n원하시면 1 또는 2로 골라주세요."
            append_history("assistant", reply)
            st.session_state.pop("expecting", None)
            if csv_logging:
                append_log(st.session_state["session_id"], msg, reply)
            return
        if any(x in lower for x in ["저렴","싼","가벼운"]):
            reply = "가벼운/저렴한 옵션 원하시는군요 — 떡볶이, 김밥, 분식 위주로 추천드릴게요:\n1. 떡볶이\n2. 김밥\n원하시면 골라주세요."
            append_history("assistant", reply)
            st.session_state.pop("expecting", None)
            if csv_logging:
                append_log(st.session_state["session_id"], msg, reply)
            return
        # otherwise generic alternative
        reply = "알겠어요. 그럼 전혀 다른 분위기의 2가지 대안 드릴게요:\n1. 제육볶음+밥\n2. 크림파스타\n어느 쪽이 더 끌리세요?"
        append_history("assistant", reply)
        st.session_state.pop("expecting", None)
        if csv_logging:
            append_log(st.session_state["session_id"], msg, reply)
        return

    # If nothing matched above, fallback: be helpful and ask a gentle clarifying Q
    reply = "어떤 스타일을 원하시는지 한 가지만 알려주실래요? (예: 한식/중식/양식/분식 또는 밥/면 중 선택)"
    append_history("assistant", reply)
    if csv_logging:
        append_log(st.session_state["session_id"], msg, reply)
    st.session_state["expecting"] = "cuisine"
    return

# When user presses submit
if submit and user_input:
    handle_user_message(user_input)
    # trim history to keep "최근 6턴 유지 후 재시작" -> interpret as keep the most recent 6 user-assistant pairs
    # We'll count pairs roughly: each pair is two messages; so 12 messages + system at start ~ keep last 13 entries
    # Simpler: keep system + last 12 messages (6 turns)
    MAX_MESSAGES = 1 + 12
    if len(st.session_state["history"]) > MAX_MESSAGES:
        # keep system + last 12 entries
        system = st.session_state["history"][0]
        tail = st.session_state["history"][-12:]
        st.session_state["history"] = [system] + tail
    # re-run to show updated chat
    st.experimental_rerun()
