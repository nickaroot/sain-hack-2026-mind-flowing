import { useEffect, useState } from "react"
import type { CSSProperties } from "react"
import { BookOpen, Check, ChevronRight, Clock, Copy, LoaderCircle, RotateCw, Square, Volume2, WifiOff } from "lucide-react"
import { Avatar } from "./Avatar"
import { answerOf } from "../chat"
import type { Turn } from "../chat"
import type { ApiError, Citation } from "../api"
import type { Persona } from "../experts"
import { useElapsed } from "../hooks"
import type { useSpeech } from "../hooks"

type Props = {
  turns: Turn[]
  personas: Persona[]
  active: Persona
  openSource: Citation | null
  onSwitch: (id: string) => void
  onRetry: (turnId: string) => void
  onSource: (citation: Citation) => void
  onRevealed: (turnId: string) => void
  speech: ReturnType<typeof useSpeech>
  registerTurn: (turnId: string, element: HTMLElement | null) => void
}

const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`

export function Thread(props: Props) {
  const { turns, active } = props
  const lastDone = [...turns].reverse().find((turn) => turn.status === "done")?.id
  return (
    <div className="thread">
      {turns.map((turn, position) => (
        <section key={turn.id} className="turn" data-last={position === turns.length - 1} ref={(element) => props.registerTurn(turn.id, element)}>
          <div className="bubble">{turn.question}</div>
          {turn.status === "pending" && <Waiting startedAt={turn.startedAt} active={active} personas={props.personas} />}
          {turn.status === "error" && turn.error && <Problem turn={turn} error={turn.error} onRetry={() => props.onRetry(turn.id)} />}
          {turn.status === "done" && turn.response && <Reply {...props} turn={turn} withSwitch={turn.id === lastDone} />}
        </section>
      ))}
    </div>
  )
}

/** Ожидание: потокового ответа нет, сервер отвечает одним куском через 10–20 секунд (до 90). */
function Waiting({ startedAt, active, personas }: { startedAt: number; active: Persona; personas: Persona[] }) {
  const seconds = useElapsed(startedAt, true)
  // Этапы показаны по времени: порядок как на сервере (поиск → чтение → ответ), но это не его отчёт.
  const status = seconds < 4 ? "Ищу подходящее исследование…" : seconds < 9 ? "Читаю раздел статьи…" : `Готовлю ответ за ${personas.length === 3 ? "всех троих" : "всех экспертов"}…`
  return (
    <div className="waiting" role="status">
      <div className="waiting-row">
        <Avatar persona={active} size={32} state="thinking" />
        <span className="dots-pulse" aria-hidden="true">
          {personas.slice(0, 4).map((persona) => (
            <i key={persona.id} style={{ "--h": persona.hue } as CSSProperties} />
          ))}
        </span>
        <span className="shimmer" key={status}>
          {status}
        </span>
        <span className="waiting-clock">{clock(seconds)}</span>
      </div>
      {seconds >= 20 && (
        <p className="note rise">Обычно это 10–20 секунд. Сейчас дольше — жду ответ до полутора минут, страницу можно не обновлять.</p>
      )}
    </div>
  )
}

function problemCopy(error: ApiError): { title: string; text: string; action: string | null; icon: typeof Clock } {
  switch (error.kind) {
    case "busy":
      return { title: "Эксперты заняты другими вопросами", text: "Ваш вопрос сохранён.", action: "Повторить сейчас", icon: Clock }
    case "quota":
      return {
        title: "На сегодня вопросы закончились",
        text: error.message || "Суточный запас разговоров исчерпан. Полученные ответы можно читать и переключать между экспертами.",
        action: null,
        icon: Clock,
      }
    case "paused":
      return { title: "Сервер сейчас занят", text: error.message || "Попробуйте через минуту.", action: "Повторить", icon: Clock }
    case "timeout":
      return {
        title: "Ответ не успел за полторы минуты",
        text: "Расчёт на сервере продолжается: повтор того же вопроса дождётся его и не потратит лимит.",
        action: "Повторить вопрос",
        icon: Clock,
      }
    case "network":
      return { title: "Нет связи с сервером", text: "Проверьте интернет и повторите вопрос.", action: "Повторить", icon: WifiOff }
    default:
      return { title: "Не получилось задать вопрос", text: error.message || "Попробуйте ещё раз.", action: "Повторить", icon: RotateCw }
  }
}

function Problem({ turn, error, onRetry }: { turn: Turn; error: ApiError; onRetry: () => void }) {
  const copy = problemCopy(error)
  const Icon = copy.icon
  // «Заняты» проходит само: отправляем вопрос снова, но не больше трёх раз подряд.
  const auto = error.kind === "busy" && turn.attempts < 4
  const wait = Math.min(60, Math.max(3, Math.round(error.retryAfter ?? 15)))
  const [left, setLeft] = useState(wait)
  useEffect(() => {
    if (!auto) return
    if (left <= 0) {
      onRetry()
      return
    }
    const timer = setTimeout(() => setLeft((value) => value - 1), 1000)
    return () => clearTimeout(timer)
  }, [auto, left, onRetry])

  return (
    <div className="problem rise" role="alert">
      <Icon size={20} className="problem-icon" aria-hidden="true" />
      <div className="problem-body">
        <strong>{copy.title}</strong>
        <span>
          {copy.text}
          {auto && ` Отправлю его снова через ${left} с.`}
          {!auto && error.kind === "paused" && error.retryAfter ? ` Сервер просит подождать ${Math.round(error.retryAfter)} с.` : ""}
        </span>
        {copy.action && (
          <button type="button" className="button-solid" onClick={onRetry}>
            {copy.action}
          </button>
        )}
      </div>
    </div>
  )
}

type ReplyProps = Props & { turn: Turn; withSwitch: boolean }

function Reply({ turn, active, personas, withSwitch, openSource, onSwitch, onRetry, onSource, onRevealed, speech }: ReplyProps) {
  const response = turn.response!
  const shown = answerOf(response, active.id)
  const speechKey = `${turn.id}:${active.id}`
  const [copied, setCopied] = useState(false)

  // Первое появление — по словам; смена эксперта — мгновенная замена текста: ответ уже был готов.
  const fresh = !!turn.fresh
  const words = shown.text.split(/\s+/).length
  const step = Math.min(14, 700 / words) * active.pace
  useEffect(() => {
    if (!fresh) return
    const timer = setTimeout(() => onRevealed(turn.id), words * step + 900)
    return () => clearTimeout(timer)
  }, [fresh, onRevealed, step, turn.id, words])

  let counter = 0
  const paragraphs = shown.text.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(shown.text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      // буфер обмена недоступен — молча ничего не делаем
    }
  }

  return (
    <article className="reply">
      <header className="reply-head">
        <Avatar persona={active} size={32} state={speech.playing === speechKey ? "speaking" : "active"} getLevel={speech.level} />
        <span>{active.name}</span>
      </header>

      <div className={fresh ? "reply-text" : "reply-text swap"} key={active.id}>
        {paragraphs.map((block, position) => (
          <p key={position}>
            {fresh
              ? block.split(/\s+/).map((word, inner) => (
                  <span key={inner} className="word" style={{ animationDelay: `${Math.round(counter++ * step)}ms` }}>
                    {word}{" "}
                  </span>
                ))
              : block}
          </p>
        ))}
      </div>

      <div className={fresh ? "reply-extras late" : "reply-extras"}>
        {shown.borrowed && <p className="note">У выбранного эксперта нет ответа на этот вопрос — показан ответ того, кого спрашивали.</p>}
        {shown.degraded && (
          <div className="degraded">
            <span>Нейросеть не успела ответить — это короткий запасной текст.</span>
            <button type="button" className="button-ghost" onClick={() => onRetry(turn.id)}>
              <RotateCw size={16} /> Спросить ещё раз
            </button>
          </div>
        )}

        {response.citations.map((citation) => {
          const open = openSource?.article_id === citation.article_id && openSource?.section_id === citation.section_id
          return (
            <button key={`${citation.article_id}/${citation.section_id}`} type="button" className="citation" data-open={open} onClick={() => onSource(citation)}>
              <BookOpen size={20} className="citation-icon" aria-hidden="true" />
              <span className="citation-body">
                <span className="citation-title" lang="en">
                  {citation.article_title}
                </span>
                <span className="citation-meta">
                  <span lang="en">{citation.section_title}</span>
                  {citation.year ? ` · ${citation.year}` : ""}
                </span>
              </span>
              {open ? <span className="citation-state">Открыт</span> : <ChevronRight size={18} className="citation-arrow" aria-hidden="true" />}
            </button>
          )
        })}

        <div className="reply-actions">
          {withSwitch && personas.length > 1 && (
            <div className="switch-block">
              <span className="note">Источник тот же — меняется подача</span>
              <div className="switch" role="radiogroup" aria-label="Тот же ответ голосом другого эксперта" data-many={personas.length > 3}>
                {personas.length <= 3 && (
                  <i
                    className="switch-thumb"
                    aria-hidden="true"
                    style={{ width: `calc((100% - 8px) / ${personas.length})`, transform: `translateX(${personas.findIndex((persona) => persona.id === active.id) * 100}%)` }}
                  />
                )}
                {personas.map((persona) => (
                  <button key={persona.id} type="button" role="radio" aria-checked={persona.id === active.id} onClick={() => onSwitch(persona.id)}>
                    {persona.short}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="reply-tools">
            <button
              type="button"
              className="icon-button"
              aria-label={speech.playing === speechKey ? "Остановить озвучивание" : "Озвучить ответ"}
              onClick={() => void speech.toggle(speechKey, shown.text, active.id)}
            >
              {speech.loading === speechKey ? <LoaderCircle size={18} className="spin" /> : speech.playing === speechKey ? <Square size={16} /> : <Volume2 size={18} />}
            </button>
            <button type="button" className="icon-button" aria-label={copied ? "Скопировано" : "Скопировать ответ"} onClick={() => void copy()}>
              {copied ? <Check size={18} /> : <Copy size={18} />}
            </button>
          </div>
        </div>
        {speech.error?.key === speechKey && (
          <p className="note" role="alert">
            {speech.error.message}
          </p>
        )}
      </div>
    </article>
  )
}
