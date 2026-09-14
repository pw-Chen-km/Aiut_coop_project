const chat = document.querySelector("#chat");
const form = document.querySelector("#composer");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const statusDot = document.querySelector("#statusDot");
const statusText = document.querySelector("#statusText");
const history = [];
let turn = 0;

function clock(date = new Date()) {
  return new Intl.DateTimeFormat("zh-TW", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(date);
}

function addMessage(role, text, { duration, citations, error = false } = {}) {
  document.querySelector(".welcome")?.remove();
  const row = document.createElement("article");
  row.className = `message ${role}${error ? " error" : ""}`;
  const bubble = document.createElement("div"); bubble.className = "bubble";
  const body = document.createElement("div"); body.className = "bubble-body"; body.textContent = text;
  const meta = document.createElement("div"); meta.className = "meta";
  const time = document.createElement("span"); time.textContent = clock(); meta.append(time);
  if (duration !== undefined) { const elapsed = document.createElement("span"); elapsed.textContent = `耗時 ${duration.toFixed(1)} 秒`; meta.append(elapsed); }
  bubble.append(body);
  if (Array.isArray(citations) && citations.length) {
    const list = document.createElement("ul"); list.className = "citations";
    citations.forEach((citation) => { const item = document.createElement("li"); item.textContent = citation.title || citation.section_title || citation.chunk_id || "來源"; list.append(item); });
    bubble.append(list);
  }
  bubble.append(meta); row.append(bubble); chat.append(row); chat.scrollTop = chat.scrollHeight; return row;
}

function addThinking() {
  const row = document.createElement("article"); row.className = "message assistant thinking";
  row.innerHTML = '<div class="bubble"><div class="bubble-body"><span>正在查找文件</span><i></i><i></i><i></i></div></div>';
  chat.append(row); chat.scrollTop = chat.scrollHeight; return row;
}

async function checkHealth() {
  try { const response = await fetch("/health/ready", { cache: "no-store" }); if (!response.ok) throw new Error(); statusDot.className = "status-dot ready"; statusText.textContent = "服務可用"; }
  catch { statusDot.className = "status-dot error"; statusText.textContent = "服務未就緒"; }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault(); const question = input.value.trim(); if (!question || send.disabled) return;
  input.value = ""; input.style.height = "auto"; addMessage("user", question);
  const priorHistory = history.slice(); history.push({ role: "user", content: question, turn_id: ++turn });
  send.disabled = true; const pending = addThinking(); const started = performance.now();
  try {
    const payload = { qid: `web-${Date.now()}`, question };
    if (priorHistory.length) payload.history = priorHistory;
    const response = await fetch("/v1/answer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const result = await response.json(); if (!response.ok || result.status !== "ok") throw new Error(result.message || result.error || "服務無法完成回答");
    pending.remove(); const answer = result.answer || "目前沒有可顯示的回答。";
    addMessage("assistant", answer, { duration: (performance.now() - started) / 1000, citations: result.citations }); history.push({ role: "assistant", content: answer, turn_id: ++turn });
  } catch (error) { pending.remove(); addMessage("assistant", `發生錯誤：${error.message}`, { duration: (performance.now() - started) / 1000, error: true }); history.pop(); turn -= 1; }
  finally { send.disabled = false; input.focus(); }
});

input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 160)}px`; });
input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
checkHealth(); input.focus();
