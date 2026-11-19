import { useEffect, useRef, useState } from "react"
import type { CSSProperties } from "react"
import { ChevronLeft, ChevronRight, LoaderCircle, Volume2 } from "lucide-react"
import { Avatar } from "./Avatar"
import { SAMPLE_QUESTION, accentStyle } from "../experts"
import type { Persona } from "../experts"
import type { useSpeech } from "../hooks"

type Props = {
  personas: Persona[]
  activeId: string
  onActivate: (id: string) => void
  onAbout: () => void
  speech: ReturnType<typeof useSpeech>
}

const sentences = (text: string, count: number) =>
  text
    .split(/(?<=[.!?…])\s+/)
    .slice(0, count)
    .join(" ")

/** Экран выбора: аватары листаются, выбранный эксперт «говорит» — отвечает на общий для всех вопрос. */
export function Gallery({ personas, activeId, onActivate, onAbout, speech }: Props) {
  const index = Math.max(0, personas.findIndex((persona) => persona.id === activeId))
  const active = personas[index]
  const line = active.sample ?? sentences(active.about, 2)
  const words = line.split(" ")
  const step = 90 * active.pace
  const key = `gallery:${active.id}`

  // «Говорит» — пока по словам появляется реплика (и пока звучит голос, если его включили).
  const [typing, setTyping] = useState(true)
  const [run, setRun] = useState(0)
  useEffect(() => {
    setTyping(true)
    const timer = setTimeout(() => setTyping(false), words.length * step + 600)
    return () => clearTimeout(timer)
  }, [active.id, run, words.length, step])

  const voiceOn = speech.playing === key
  const voiceLoading = speech.loading === key
  const speaking = typing || voiceOn

  const go = (offset: number) => {
    const next = personas[Math.min(personas.length - 1, Math.max(0, index + offset))]
    if (next.id !== active.id) {
      speech.stop()
      onActivate(next.id)
    }
  }

  const drag = useRef<number | null>(null)
  const listen = () => {
    if (!voiceOn && !voiceLoading) setRun((value) => value + 1)
    void speech.toggle(key, line, active.id)
  }

  return (
    <section className="gallery" aria-label="Выбор эксперта">
      <div className="gallery-head">
        <h1>Кто ответит на ваш вопрос о сне?</h1>
        <p>Листайте: все читают одни и те же исследования, но объясняют их по-разному.</p>
      </div>

      <div
        className="stage"
        onPointerDown={(event) => (drag.current = event.clientX)}
        onPointerUp={(event) => {
          const delta = drag.current === null ? 0 : event.clientX - drag.current
          drag.current = null
          if (Math.abs(delta) > 40) go(delta < 0 ? 1 : -1)
        }}
        onPointerCancel={() => (drag.current = null)}
        onKeyDown={(event) => {
          if (event.key === "ArrowRight") go(1)
          if (event.key === "ArrowLeft") go(-1)
        }}
      >
        {personas.map((persona, position) => {
          const offset = position - index
          const style = { ...accentStyle(persona.hue), "--k": offset, zIndex: offset === 0 ? 3 : 1 } as CSSProperties
          return (
            <button
              key={persona.id}
              type="button"
              className="slide"
              data-pos={offset === 0 ? "center" : Math.abs(offset) === 1 ? "near" : "far"}
              style={style}
              aria-label={persona.name}
              aria-pressed={offset === 0}
              onClick={() => offset !== 0 && go(offset)}
            >
              <Avatar persona={persona} size={208} state={offset !== 0 ? "idle" : speaking ? "speaking" : "active"} getLevel={voiceOn ? speech.level : undefined} />
            </button>
          )
        })}
        <button type="button" className="icon-button stage-arrow" data-side="left" aria-label="Предыдущий эксперт" disabled={index === 0} onClick={() => go(-1)}>
          <ChevronLeft size={20} />
        </button>
        <button
          type="button"
          className="icon-button stage-arrow"
          data-side="right"
          aria-label="Следующий эксперт"
          disabled={index === personas.length - 1}
          onClick={() => go(1)}
        >
          <ChevronRight size={20} />
        </button>
      </div>

      <button type="button" className="voice-pill" onClick={listen} aria-label={voiceOn ? "Остановить голос" : `Послушать, как говорит ${active.name}`}>
        {voiceLoading ? (
          <>
            <LoaderCircle size={16} className="spin" /> Готовлю голос…
          </>
        ) : speaking ? (
          <>
            <span className="bars" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </span>
            Говорит
          </>
        ) : (
          <>
            <Volume2 size={16} /> Послушать голос
          </>
        )}
      </button>
      {speech.error?.key === key && <p className="note">{speech.error.message}</p>}

      <div className="gallery-name" key={`name-${active.id}`}>
        <h2>{active.name}</h2>
        <p>{active.tagline}</p>
      </div>

      <div className="quote">
        <span className="quote-label">{active.sample ? `На вопрос «${SAMPLE_QUESTION}» отвечает так` : "Как думает"}</span>
        <p className="quote-text" key={`${active.id}-${run}`} aria-live="polite">
          {words.map((word, position) => (
            <span key={position} className="word" style={{ animationDelay: `${Math.round(position * step)}ms` }}>
              {word}{" "}
            </span>
          ))}
        </p>
      </div>

      <div className="traits" key={`traits-${active.id}`}>
        {active.traits.map((trait) => (
          <span key={trait} className="chip-accent">
            {trait}
          </span>
        ))}
        <button type="button" className="link-button" onClick={onAbout}>
          Подробнее
        </button>
      </div>

      <div className="dots" aria-hidden="true">
        {personas.map((persona) => (
          <i key={persona.id} data-on={persona.id === active.id} />
        ))}
      </div>
    </section>
  )
}
