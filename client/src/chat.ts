// Модель разговора в клиенте: вопрос, его состояние и ответ сразу всех экспертов.
import type { Answer, ApiError, ChatResponse, HistoryItem } from "./api"
import { MAX_TEXT } from "./api"

export type Turn = {
  id: string
  question: string
  status: "pending" | "done" | "error"
  /** Когда вопрос ушёл на сервер: от этого времени идёт таймер ожидания. */
  startedAt: number
  attempts: number
  response?: ChatResponse
  error?: ApiError
  /** Ответ только что пришёл и ещё не показан: появится фразами. */
  fresh?: boolean
}

export type Shown = Answer & {
  /** Запасной текст вместо ответа нейросети — стоит предложить повторить вопрос. */
  degraded: boolean
  /** У выбранного эксперта нет ответа на этот вопрос — показан ответ того, кого спрашивали. */
  borrowed: boolean
}

/** Ответ на вопрос голосом выбранного эксперта. Сервер прислал всех сразу — запрос не нужен. */
export function answerOf(response: ChatResponse, expertId: string): Shown {
  const own = response.answers?.[expertId]
  if (own) return { ...own, degraded: response.degraded && response.expert_id === expertId, borrowed: false }
  const asked = response.answers?.[response.expert_id] ?? { text: response.text, refused: response.refused }
  return { ...asked, degraded: response.degraded, borrowed: response.expert_id !== expertId }
}

/** Предыдущие реплики разговора для сервера: от старых к новым, не больше 20, каждая до 2000 знаков. */
export function historyFor(turns: Turn[], expertId: string): HistoryItem[] {
  const items: HistoryItem[] = []
  for (const turn of turns) {
    if (turn.status !== "done" || !turn.response) continue
    items.push({ role: "user", text: turn.question.slice(0, MAX_TEXT) })
    items.push({ role: "assistant", text: answerOf(turn.response, expertId).text.slice(0, MAX_TEXT) })
  }
  return items.slice(-20)
}
