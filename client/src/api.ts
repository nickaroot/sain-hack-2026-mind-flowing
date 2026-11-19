// Слой общения с серверной частью. Договор — в client/README.md.
import * as mocks from "./mocks"

export type Expert = { expert_id: string; name: string; description: string; speech_style: string }
export type Citation = {
  article_id: string
  section_id: string
  article_title: string
  section_title: string
  year?: number | null
  doi?: string | null
}
export type Answer = { text: string; refused: boolean }
export type ChatResponse = {
  expert_id: string
  text: string
  refused: boolean
  degraded: boolean
  answers: Record<string, Answer>
  citations: Citation[]
}
export type Source = Citation & { text: string }
export type HistoryItem = { role: "user" | "assistant"; text: string }
export type ExpertsResponse = { experts: Expert[]; user_id: string | null }

/** Что случилось — в терминах того, что показать человеку. */
export type ApiErrorKind =
  | "busy" // 429: слишком много вопросов сразу, скоро освободится
  | "quota" // 429: дневной запас кончился
  | "paused" // 503: сервер занят или действует пауза
  | "timeout" // 504 или клиент не дождался: расчёт продолжается, повтор его дождётся
  | "network" // нет связи
  | "invalid" // 413/415/422: что-то не так с самим запросом
  | "voice" // 502: сервис голоса недоступен
  | "unknown"

export class ApiError extends Error {
  kind: ApiErrorKind
  status: number
  /** Секунды из заголовка Retry-After, если сервер его прислал. */
  retryAfter: number | null
  constructor(kind: ApiErrorKind, message: string, status = 0, retryAfter: number | null = null) {
    super(message)
    this.kind = kind
    this.status = status
    this.retryAfter = retryAfter
  }
}

export const API_URL = (import.meta.env.VITE_API_URL || "https://team23.hack.be-healthy.tech").replace(/\/+$/, "")
const params = new URLSearchParams(location.search)
/** Заглушки: сборка `npm run mock` или `?mock` в адресе. Стенд в этом режиме не вызывается вообще. */
export const MOCK = __MOCK__ || params.has("mock")
const USER_ID = import.meta.env.VITE_USER_ID || params.get("user") || null

export const MAX_TEXT = 2000
const CHAT_TIMEOUT_MS = 100_000 // сервер отвечает не позже чем через 90 с
const SHORT_TIMEOUT_MS = 30_000

function classify(status: number, detail: string, retryAfter: number | null): ApiErrorKind {
  if (status === 429) {
    const daily = (retryAfter ?? 0) >= 600 || /сегодня|завтра|исчерпан|закончил/i.test(detail)
    return daily ? "quota" : "busy"
  }
  if (status === 503) return "paused"
  if (status === 504) return "timeout"
  if (status === 502) return "voice"
  if (status === 413 || status === 415 || status === 422) return "invalid"
  return "unknown"
}

async function request(path: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  let response: Response
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal })
  } catch (error) {
    if (controller.signal.aborted) throw new ApiError("timeout", "Ответ не пришёл вовремя", 0)
    throw new ApiError("network", error instanceof Error ? error.message : "Нет связи", 0)
  } finally {
    clearTimeout(timer)
  }
  if (response.ok) return response
  let detail = ""
  try {
    const body = await response.json()
    detail = typeof body?.detail === "string" ? body.detail : ""
  } catch {
    // тело не JSON — покажем общий текст
  }
  const header = Number(response.headers.get("Retry-After"))
  const retryAfter = Number.isFinite(header) && header > 0 ? header : null
  throw new ApiError(classify(response.status, detail, retryAfter), detail, response.status, retryAfter)
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
})

export async function getExperts(): Promise<ExpertsResponse> {
  if (MOCK) return mocks.experts()
  return (await request("/api/experts", {}, SHORT_TIMEOUT_MS)).json()
}

export async function chat(expertId: string, text: string, history: HistoryItem[]): Promise<ChatResponse> {
  if (MOCK) return mocks.chat(expertId, text)
  const body: Record<string, unknown> = { expert_id: expertId, text, history }
  if (USER_ID) body.user_id = USER_ID
  return (await request("/api/chat", json(body), CHAT_TIMEOUT_MS)).json()
}

export async function getSource(articleId: string, sectionId: string): Promise<Source> {
  if (MOCK) return mocks.source(articleId, sectionId)
  const path = `/api/sources/${encodeURIComponent(articleId)}/${encodeURIComponent(sectionId)}`
  return (await request(path, {}, SHORT_TIMEOUT_MS)).json()
}

export async function transcribe(recording: Blob): Promise<string> {
  if (MOCK) return mocks.transcribe()
  const init: RequestInit = { method: "POST", headers: { "Content-Type": recording.type || "audio/webm" }, body: recording }
  const data = await (await request("/api/transcribe", init, SHORT_TIMEOUT_MS)).json()
  return typeof data?.text === "string" ? data.text : ""
}

/** mp3 с голосом эксперта. На заглушках возвращает null: тогда говорит синтезатор браузера. */
export async function speak(text: string, expertId: string): Promise<Blob | null> {
  if (MOCK) return mocks.speak()
  const response = await request("/api/speak", json({ text: text.slice(0, MAX_TEXT), expert_id: expertId }), SHORT_TIMEOUT_MS)
  return response.blob()
}
