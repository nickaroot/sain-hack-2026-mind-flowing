import { useEffect, useRef } from "react"
import { ArrowUp, LoaderCircle, Mic, Square, X } from "lucide-react"
import { MAX_TEXT } from "../api"
import { useElapsed } from "../hooks"
import type { useRecorder } from "../hooks"

type Props = {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  /** Ответ на предыдущий вопрос ещё готовится: набирать можно, отправлять нельзя. */
  pending: boolean
  placeholder: string
  recorder: ReturnType<typeof useRecorder>
}

const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`
const BARS = 28

export function Composer({ value, onChange, onSubmit, pending, placeholder, recorder }: Props) {
  const field = useRef<HTMLTextAreaElement>(null)
  const seconds = useElapsed(recorder.startedAt, recorder.state === "recording")
  const canSend = value.trim().length > 0 && !pending

  // Поле растёт вместе с текстом до шести строк. Пустое поле всегда в одну строку: подсказка на высоту
  // не влияет, а при смене ширины окна высота пересчитывается, а не остаётся от прежней ширины.
  useEffect(() => {
    const element = field.current
    if (!element) return
    const fit = () => {
      element.style.height = "auto"
      element.style.height = value ? `${Math.min(element.scrollHeight, 160)}px` : ""
    }
    fit()
    window.addEventListener("resize", fit)
    return () => window.removeEventListener("resize", fit)
  }, [value, recorder.state])

  if (recorder.state !== "idle") {
    // Диктовка: волна и время, справа — «стоп» (текст в поле, его можно поправить) и «отправить» (сразу в разговор).
    const busy = recorder.state === "transcribing"
    return (
      <div className="composer-wrap">
        <div className="composer" data-mode="record">
          <button type="button" className="icon-button" aria-label="Отменить запись" onClick={recorder.cancel} disabled={busy}>
            <X size={20} />
          </button>
          <span className="wave" aria-hidden="true">
            {Array.from({ length: BARS }, (_, position) => {
              // Форма волны постоянная, высота — от громкости микрофона.
              const shape = 0.35 + 0.65 * Math.abs(Math.sin(position * 1.7))
              return <i key={position} style={{ transform: `scaleY(${busy ? 0.15 : Math.max(0.12, shape * recorder.level)})` }} />
            })}
          </span>
          <span className="record-status" role="status">
            {busy ? "Распознаю…" : clock(seconds)}
          </span>
          <button type="button" className="icon-button soft" aria-label="Остановить: текст появится в поле" onClick={() => void recorder.finish()} disabled={busy}>
            {busy && recorder.ending === "insert" ? <LoaderCircle size={20} className="spin" /> : <Square size={14} fill="currentColor" />}
          </button>
          <button
            type="button"
            className="icon-button send"
            aria-label={pending ? "Ответ готовится: отправить можно будет после него" : "Отправить вопрос"}
            onClick={() => void recorder.send()}
            disabled={busy || pending}
          >
            {busy && recorder.ending === "send" ? <LoaderCircle size={20} className="spin" /> : <ArrowUp size={20} strokeWidth={2.4} />}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          if (canSend) onSubmit()
        }}
      >
        <label className="sr-only" htmlFor="question">
          Вопрос эксперту
        </label>
        <textarea
          id="question"
          ref={field}
          rows={1}
          value={value}
          maxLength={MAX_TEXT}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            // На клавиатуре Enter отправляет, Shift+Enter переносит строку; на телефоне Enter — перенос.
            const touch = matchMedia("(pointer: coarse)").matches
            if (event.key === "Enter" && !event.shiftKey && !touch && !event.nativeEvent.isComposing) {
              event.preventDefault()
              if (canSend) onSubmit()
            }
          }}
        />
        {recorder.supported && (
          <button type="button" className="icon-button" aria-label="Задать вопрос голосом" onClick={() => void recorder.start()}>
            <Mic size={20} />
          </button>
        )}
        <button type="submit" className="icon-button send" aria-label={pending ? "Ответ готовится" : "Отправить"} disabled={!canSend}>
          {pending ? <LoaderCircle size={20} className="spin" /> : <ArrowUp size={20} strokeWidth={2.4} />}
        </button>
      </form>
      {value.length > MAX_TEXT - 200 && (
        <p className="note" role="status">
          {value.length} из {MAX_TEXT} знаков
        </p>
      )}
      {recorder.error && (
        <p className="note" role="alert">
          {recorder.error}
        </p>
      )}
    </div>
  )
}
