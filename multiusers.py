"""멀티유저/멀티세션 RAG 챗봇 — DB user 테이블 기반 로그인 + Supabase 저장."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from postgrest.exceptions import APIError
from supabase import Client, create_client

# DB FK(chat_sessions, chat_messages)가 public.user 를 참조합니다.
USER_TABLE = "user"

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = APP_DIR / "logs"

load_dotenv(dotenv_path=ENV_PATH)

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_BATCH_SIZE = 10
RAG_MATCH_COUNT = 10
PBKDF2_ITERATIONS = 120_000

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()


def _config_value(key: str) -> str:
    """st.secrets 우선, 없으면 os.getenv."""
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except (FileNotFoundError, AttributeError, KeyError):
        pass
    try:
        general = st.secrets.get("general", {})
        if isinstance(general, dict) and key in general:
            return str(general[key]).strip()
    except (FileNotFoundError, AttributeError, KeyError):
        pass
    return os.getenv(key, "").strip()


def _env_keys() -> tuple[str, str, str]:
    return (
        _config_value("OPENAI_API_KEY"),
        _config_value("SUPABASE_URL"),
        _config_value("SUPABASE_ANON_KEY"),
    )


def _missing_key_message() -> str | None:
    openai_key, supabase_url, supabase_key = _env_keys()
    missing: list[str] = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not supabase_key:
        missing.append("SUPABASE_ANON_KEY")
    if not missing:
        return None
    src = "Streamlit Secrets 또는 `.env`"
    return (
        "# 환경 변수 안내\n\n"
        f"{src}에 다음 키를 설정해 주세요.\n\n"
        + "\n".join(f"- **{k}**" for k in missing)
    )


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected_hex = stored_hash.split("$", 1)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            PBKDF2_ITERATIONS,
        )
        return secrets.compare_digest(digest.hex(), expected_hex)
    except (ValueError, AttributeError):
        return False


@st.cache_resource
def get_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)


def get_llm() -> ChatOpenAI:
    key = _env_keys()[0]
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model=LLM_MODEL, temperature=0.7, api_key=key)


def get_embeddings() -> OpenAIEmbeddings:
    key = _env_keys()[0]
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=key)


def _current_user_id() -> str | None:
    if not st.session_state.get("logged_in"):
        return None
    return st.session_state.get("user_id")


def _db_setup_hint() -> str:
    return (
        "Supabase Dashboard → SQL Editor에서 "
        "`multi-users-migrate.sql` 실행 후 브라우저를 새로고침해 주세요. "
        "(파일 위치: 7.MultiService/code/)"
    )


def register_user(client: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력해 주세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    try:
        existing = (
            client.table(USER_TABLE)
            .select("id")
            .eq("login_id", login_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            return False, "이미 사용 중인 아이디입니다."

        client.table(USER_TABLE).insert(
            {
                "login_id": login_id,
                "password_hash": hash_password(password),
            }
        ).execute()
    except APIError as exc:
        err = str(exc)
        if "PGRST205" in err or "Could not find the table" in err:
            return False, f"회원 테이블이 없습니다. {_db_setup_hint()}"
        logger.warning("register_user failed: %s", exc)
        return False, f"회원가입 실패: {exc}"
    return True, "회원가입이 완료되었습니다. 로그인해 주세요."


def login_user(client: Client, login_id: str, password: str) -> tuple[bool, str, str | None]:
    login_id = login_id.strip()
    try:
        resp = (
            client.table(USER_TABLE)
            .select("id, password_hash")
            .eq("login_id", login_id)
            .limit(1)
            .execute()
        )
    except APIError as exc:
        err = str(exc)
        if "PGRST205" in err or "Could not find the table" in err:
            return False, f"회원 테이블이 없습니다. {_db_setup_hint()}", None
        return False, f"로그인 실패: {exc}", None

    rows = resp.data or []
    if not rows:
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None

    row = rows[0]
    if not verify_password(password, row["password_hash"]):
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None

    return True, "로그인되었습니다.", str(row["id"])


def _session_owned_by_user(client: Client, session_id: str, user_id: str) -> bool:
    resp = (
        client.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def fetch_all_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    resp = (
        client.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(resp.data or [])


def fetch_session_messages(
    client: Client, user_id: str, session_id: str
) -> list[dict[str, str]]:
    if not _session_owned_by_user(client, session_id, user_id):
        return []
    resp = (
        client.table("chat_messages")
        .select("role, content, message_order")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("message_order")
        .execute()
    )
    return [{"role": r["role"], "content": r["content"]} for r in (resp.data or [])]


def _upsert_session_row(
    client: Client, user_id: str, session_id: str, title: str
) -> None:
    now = datetime.utcnow().isoformat()
    client.table("chat_sessions").upsert(
        {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "updated_at": now,
        },
        on_conflict="id",
    ).execute()


def save_messages_to_db(
    client: Client,
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
) -> None:
    if not _session_owned_by_user(client, session_id, user_id):
        return
    client.table("chat_messages").delete().eq("session_id", session_id).eq(
        "user_id", user_id
    ).execute()
    if not messages:
        return
    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
            "message_order": idx,
        }
        for idx, m in enumerate(messages)
    ]
    client.table("chat_messages").insert(rows).execute()


def insert_new_session(
    client: Client,
    user_id: str,
    title: str,
    messages: list[dict[str, str]],
    source_session_id: str | None,
) -> str:
    new_id = str(uuid.uuid4())
    client.table("chat_sessions").insert(
        {"id": new_id, "user_id": user_id, "title": title}
    ).execute()

    if messages:
        save_messages_to_db(client, user_id, new_id, messages)

    if source_session_id and _session_owned_by_user(
        client, source_session_id, user_id
    ):
        _copy_vectors_to_session(client, source_session_id, new_id)

    return new_id


def delete_session_from_db(client: Client, user_id: str, session_id: str) -> None:
    client.table("chat_sessions").delete().eq("id", session_id).eq(
        "user_id", user_id
    ).execute()


def auto_save_current_session(client: Client, user_id: str) -> None:
    sid = st.session_state.get("session_id")
    if not sid:
        return
    title = st.session_state.get("session_title") or "새 세션"
    _upsert_session_row(client, user_id, sid, title)
    save_messages_to_db(
        client, user_id, sid, st.session_state.conversation_memory
    )


def _copy_vectors_to_session(
    client: Client,
    from_session_id: str,
    to_session_id: str,
) -> None:
    resp = (
        client.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("session_id", from_session_id)
        .execute()
    )
    rows = resp.data or []
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(
            {
                "session_id": to_session_id,
                "file_name": row["file_name"],
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "embedding": row["embedding"],
            }
        )
        if len(batch) >= VECTOR_BATCH_SIZE:
            client.table("vector_documents").insert(batch).execute()
            batch = []
    if batch:
        client.table("vector_documents").insert(batch).execute()


def store_vectors_for_file(
    client: Client,
    session_id: str,
    file_name: str,
    splits: list[Document],
    embeddings: OpenAIEmbeddings,
) -> int:
    if not splits:
        return 0

    texts = [d.page_content for d in splits]
    vectors = embeddings.embed_documents(texts)
    stored = 0
    batch: list[dict[str, Any]] = []

    for doc, vec in zip(splits, vectors):
        batch.append(
            {
                "session_id": session_id,
                "file_name": file_name,
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "embedding": vec,
            }
        )
        if len(batch) >= VECTOR_BATCH_SIZE:
            client.table("vector_documents").insert(batch).execute()
            stored += len(batch)
            batch = []

    if batch:
        client.table("vector_documents").insert(batch).execute()
        stored += len(batch)

    return stored


def fetch_vector_file_names(client: Client, session_id: str) -> list[str]:
    resp = (
        client.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = {r["file_name"] for r in (resp.data or []) if r.get("file_name")}
    return sorted(names)


def session_has_vectors(client: Client, session_id: str) -> bool:
    resp = (
        client.table("vector_documents")
        .select("id")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def match_documents_rpc(
    client: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = RAG_MATCH_COUNT,
) -> list[Document]:
    query_vec = embeddings.embed_query(query)
    try:
        resp = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_vec,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        rows = resp.data or []
        return [
            Document(
                page_content=row.get("content", ""),
                metadata={
                    "file_name": row.get("file_name", ""),
                    "similarity": row.get("similarity"),
                    **(row.get("metadata") or {}),
                },
            )
            for row in rows
            if row.get("content")
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC match_vector_documents failed: %s", exc)
        return _match_documents_fallback(client, session_id, query, embeddings, k)


def _match_documents_fallback(
    client: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int,
) -> list[Document]:
    resp = (
        client.table("vector_documents")
        .select("content, file_name, metadata, embedding")
        .eq("session_id", session_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return []

    q_vec = embeddings.embed_query(query)

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        emb = row.get("embedding")
        if not emb or isinstance(emb, str):
            continue
        scored.append((cosine(q_vec, emb), row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        Document(
            page_content=row.get("content", ""),
            metadata={"file_name": row.get("file_name", "")},
        )
        for _, row in scored[:k]
    ]


def _process_pdf_uploads(
    client: Client,
    session_id: str,
    uploaded_files: list[Any],
    embeddings: OpenAIEmbeddings,
) -> list[str]:
    if not uploaded_files:
        return []

    processed: list[str] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for uf in uploaded_files:
        file_name = uf.name or "unknown.pdf"
        suffix = Path(file_name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()
            for d in raw_docs:
                d.metadata["file_name"] = file_name
            splits = splitter.split_documents(raw_docs)
            for d in splits:
                d.metadata["file_name"] = file_name
            if store_vectors_for_file(client, session_id, file_name, splits, embeddings):
                processed.append(file_name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return processed


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if m.get("role") == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str, context: str, memory_text: str
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{answer[:8000]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = remove_separators(str(getattr(out, "content", "") or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _generate_session_title(llm: ChatOpenAI, first_q: str, first_a: str) -> str:
    prompt = (
        "다음 첫 질문과 답변을 15자 이내 한국어 제목 한 줄로 요약하세요. "
        "따옴표·설명·줄바꿈 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{first_q[:500]}\n\n[답변]\n{first_a[:800]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = remove_separators(str(getattr(out, "content", "") or "")).strip()
        return title[:80] if title else "새 세션"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Session title generation failed: %s", exc)
        return first_q[:30] or "새 세션"


def _first_qa(messages: list[dict[str, str]]) -> tuple[str, str]:
    first_q, first_a = "", ""
    for m in messages:
        if m["role"] == "user" and not first_q:
            first_q = m["content"]
        elif m["role"] == "assistant" and first_q and not first_a:
            first_a = m["content"]
            break
    return first_q, first_a


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "logged_in": False,
        "user_id": None,
        "login_id": None,
        "chat_history": [],
        "conversation_memory": [],
        "processed_names": [],
        "session_id": None,
        "session_title": "새 세션",
        "session_list": [],
        "selected_session_label": None,
        "supabase_ready": False,
        "title_finalized": False,
        "auth_mode": "login",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _ensure_session_id(client: Client, user_id: str) -> str:
    sid = st.session_state.get("session_id")
    if sid and _session_owned_by_user(client, sid, user_id):
        return sid
    sid = str(uuid.uuid4())
    st.session_state.session_id = sid
    st.session_state.session_title = "새 세션"
    st.session_state.title_finalized = False
    _upsert_session_row(client, user_id, sid, st.session_state.session_title)
    return sid


def _apply_loaded_session(
    client: Client, user_id: str, session_id: str, title: str
) -> None:
    if not _session_owned_by_user(client, session_id, user_id):
        st.warning("해당 세션에 접근할 수 없습니다.")
        return
    messages = fetch_session_messages(client, user_id, session_id)
    st.session_state.session_id = session_id
    st.session_state.session_title = title
    st.session_state.chat_history = list(messages)
    st.session_state.conversation_memory = list(messages)
    st.session_state.processed_names = fetch_vector_file_names(client, session_id)
    st.session_state.title_finalized = True


def _clear_ui_only() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_names = []
    st.session_state.session_id = None
    st.session_state.session_title = "새 세션"
    st.session_state.title_finalized = False


def _session_label(s: dict[str, Any]) -> str:
    title = s.get("title") or "제목 없음"
    return f"{title} ({str(s.get('id', ''))[:8]})"


def _resolve_session_id_from_label(
    label: str, sessions: list[dict[str, Any]]
) -> str | None:
    for s in sessions:
        if _session_label(s) == label:
            return str(s["id"])
    return None


def _logout() -> None:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.login_id = None
    _clear_ui_only()
    st.session_state.session_list = []
    st.session_state.supabase_ready = False
    st.session_state.selected_session_label = None


def _render_auth(client: Client) -> None:
    st.markdown("### 로그인 / 회원가입")
    mode = st.radio("모드", ("로그인", "회원가입"), horizontal=True, key="auth_mode_radio")
    login_id = st.text_input("아이디 (login_id)", key="auth_login_id")
    password = st.text_input("비밀번호", type="password", key="auth_password")

    if mode == "회원가입":
        if st.button("회원가입", use_container_width=True):
            ok, msg = register_user(client, login_id, password)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    else:
        if st.button("로그인", use_container_width=True):
            ok, msg, uid = login_user(client, login_id, password)
            if ok and uid:
                st.session_state.logged_in = True
                st.session_state.user_id = uid
                st.session_state.login_id = login_id.strip()
                st.session_state.supabase_ready = False
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


def _render_header() -> None:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def _render_sidebar(client: Client, user_id: str) -> None:
    with st.sidebar:
        st.markdown(f"**로그인:** `{st.session_state.login_id}`")
        if st.button("로그아웃", use_container_width=True):
            _logout()
            st.rerun()

        st.markdown("### 세션 관리")
        st.session_state.session_list = fetch_all_sessions(client, user_id)
        labels = [_session_label(s) for s in st.session_state.session_list]

        if labels:
            current_label = None
            for s in st.session_state.session_list:
                if str(s["id"]) == str(st.session_state.session_id):
                    current_label = _session_label(s)
                    break
            idx = labels.index(current_label) if current_label in labels else 0
            selected_label = st.selectbox("저장된 세션", labels, index=idx)
            if selected_label != st.session_state.get("selected_session_label"):
                st.session_state.selected_session_label = selected_label
                load_id = _resolve_session_id_from_label(
                    selected_label, st.session_state.session_list
                )
                if load_id:
                    title = next(
                        (
                            s.get("title", "새 세션")
                            for s in st.session_state.session_list
                            if str(s["id"]) == load_id
                        ),
                        "새 세션",
                    )
                    _apply_loaded_session(client, user_id, load_id, title)
                    st.rerun()
        else:
            st.caption("저장된 세션이 없습니다.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("세션저장", use_container_width=True):
                msgs = st.session_state.conversation_memory
                if not msgs:
                    st.warning("저장할 대화가 없습니다.")
                else:
                    try:
                        llm = get_llm()
                        fq, fa = _first_qa(msgs)
                        title = (
                            _generate_session_title(llm, fq, fa)
                            if fq and fa
                            else st.session_state.session_title
                        )
                        src = st.session_state.session_id
                        new_id = insert_new_session(
                            client, user_id, title, msgs, src
                        )
                        st.session_state.session_list = fetch_all_sessions(
                            client, user_id
                        )
                        _apply_loaded_session(client, user_id, new_id, title)
                        st.success(f"세션이 저장되었습니다: {title}")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 저장 실패: {exc}")

        with col_b:
            if st.button("세션로드", use_container_width=True):
                label = st.session_state.get("selected_session_label")
                if not label and labels:
                    label = labels[0]
                load_id = (
                    _resolve_session_id_from_label(
                        label, st.session_state.session_list
                    )
                    if label
                    else None
                )
                if not load_id:
                    st.warning("불러올 세션을 선택해 주세요.")
                else:
                    title = next(
                        (
                            s.get("title", "새 세션")
                            for s in st.session_state.session_list
                            if str(s["id"]) == load_id
                        ),
                        "새 세션",
                    )
                    _apply_loaded_session(client, user_id, load_id, title)
                    st.success(f"세션을 불러왔습니다: {title}")
                    st.rerun()

        col_c, col_d = st.columns(2)
        with col_c:
            if st.button("세션삭제", use_container_width=True):
                sid = st.session_state.session_id
                if not sid:
                    st.warning("삭제할 세션이 없습니다.")
                else:
                    try:
                        delete_session_from_db(client, user_id, sid)
                        _clear_ui_only()
                        st.session_state.session_list = fetch_all_sessions(
                            client, user_id
                        )
                        if st.session_state.session_list:
                            latest = st.session_state.session_list[0]
                            _apply_loaded_session(
                                client,
                                user_id,
                                str(latest["id"]),
                                latest.get("title") or "새 세션",
                            )
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 삭제 실패: {exc}")

        with col_d:
            if st.button("화면초기화", use_container_width=True):
                _clear_ui_only()
                st.rerun()

        if st.button("vectordb", use_container_width=True):
            sid = st.session_state.session_id or _ensure_session_id(
                client, user_id
            )
            if _session_owned_by_user(client, sid, user_id):
                names = fetch_vector_file_names(client, sid)
                if names:
                    st.markdown("**Vector DB 파일 목록**")
                    for n in names:
                        st.text(f"- {n}")
                else:
                    st.info("현재 세션에 저장된 벡터 문서가 없습니다.")

        st.markdown("---")
        st.markdown(f"**모델:** `{LLM_MODEL}`")

        uploads = st.file_uploader(
            "PDF 파일 업로드", type=["pdf"], accept_multiple_files=True
        )
        if st.button("파일 처리하기"):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    sid = _ensure_session_id(client, user_id)
                    emb = get_embeddings()
                    names = _process_pdf_uploads(
                        client, sid, list(uploads), emb
                    )
                    existing = set(st.session_state.processed_names)
                    for n in names:
                        existing.add(n)
                    st.session_state.processed_names = sorted(existing)
                    auto_save_current_session(client, user_id)
                    st.success("PDF 처리 및 벡터 저장이 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        sid = st.session_state.session_id
        has_vec = (
            session_has_vectors(client, sid)
            if sid and _session_owned_by_user(client, sid, user_id)
            else False
        )
        st.text(
            f"현재 세션: {st.session_state.session_title}\n"
            f"벡터 DB: {'있음' if has_vec else '없음'}\n"
            f"대화 메시지 수: {len(st.session_state.conversation_memory)}"
        )


def main() -> None:
    st.set_page_config(
        page_title="재정경제부 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    _init_session()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    missing_msg = _missing_key_message()
    if missing_msg:
        st.markdown(missing_msg)
        return

    _, supabase_url, supabase_key = _env_keys()
    client = get_supabase_client(supabase_url, supabase_key)

    if not st.session_state.logged_in:
        _render_header()
        _render_auth(client)
        return

    user_id = _current_user_id()
    if not user_id:
        _logout()
        st.rerun()
        return

    if not st.session_state.supabase_ready:
        try:
            sessions = fetch_all_sessions(client, user_id)
            st.session_state.session_list = sessions
            if sessions and not st.session_state.session_id:
                latest = sessions[0]
                _apply_loaded_session(
                    client,
                    user_id,
                    str(latest["id"]),
                    latest.get("title") or "새 세션",
                )
            st.session_state.supabase_ready = True
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"Supabase 연결 실패. `multi-users-migrate.sql` 또는 `multi-users.sql`을 "
                f"Supabase SQL Editor에서 실행했는지 확인해 주세요.\n\n`{exc}`"
            )
            return

    _render_header()
    _render_sidebar(client, user_id)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    sid = _ensure_session_id(client, user_id)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            llm = get_llm()
            emb = get_embeddings()
            if not session_has_vectors(client, sid):
                full_answer = (
                    "# 안내\n\n"
                    "문서 기반 답변을 하려면 PDF를 업로드한 뒤 **파일 처리하기**를 눌러 주세요."
                )
                placeholder.markdown(remove_separators(full_answer))
            else:
                mem_txt = _format_memory_block(
                    st.session_state.conversation_memory[:-1]
                )
                docs = match_documents_rpc(client, sid, user_input, emb)
                context = "\n\n".join(d.page_content for d in docs) or "(관련 문서 없음)"
                messages = _build_rag_messages(user_input, context, mem_txt)

                acc = ""
                for chunk in llm.stream(messages):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
                placeholder.markdown(full_answer)

                follow = _generate_followup_section(llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_answer}
    )
    st.session_state.conversation_memory.append(
        {"role": "assistant", "content": full_answer}
    )

    try:
        fq, fa = _first_qa(st.session_state.conversation_memory)
        if fq and fa and not st.session_state.get("title_finalized"):
            st.session_state.session_title = _generate_session_title(
                get_llm(), fq, fa
            )
            st.session_state.title_finalized = True
        auto_save_current_session(client, user_id)
        st.session_state.session_list = fetch_all_sessions(client, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 저장 실패: %s", exc)


if __name__ == "__main__":
    main()
