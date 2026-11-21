import { useEffect, useRef, useState } from "react"
import type { CSSProperties } from "react"
import { ChevronLeft, ChevronRight, LoaderCircle, Play, Square } from "lucide-react"
import { Avatar } from "./Avatar"
import { SAMPLE_QUESTION, accentStyle, galleryLine } from "../experts"
import type { Persona } from "../experts"
import type { Speech } from "../speech"
import { TypedText } from "./TypedText"

type Props = {
  personas: Persona[]
  activeId: string
  onActivate: (id: string) => void
  onAbout: () => void
  speech: Speech
}

/** Экран выбора: аватары листаются, выбранный эксперт «говорит» — отвечает на общий для всех вопрос. */
export function Gallery({ personas, activeId, onActivate, onAbout, speech }: Props) {
  const index = Math.max(0, personas.findIndex((persona) => persona.id === activeId))
  const active = personas[index]
  const line = galleryLine(active)
  const key = `gallery:${active.id}`

  const voiceOn = speech.playing === key
  const voiceLoading = speech.loading === key
  // При открытии страницы эксперт молчит и текста нет — их включает кнопка «играть» в карточке реплики или
  // нажатие на самого эксперта. Текст появляется сразу и в своём темпе, голос его не задерживает.
  // После первого прослушивания выбранный эксперт отвечает сам: перелистывание — тоже нажатие, звук браузер уже разрешил.
  const waitingTap = speech.blocked
  const narrate = speech.narrate
  const follow = speech.auto && speech.started
  useEffect(() => {
    if (follow) void narrate(key, line, active.id)
    // Только при смене эксперта: первое включение делает кнопка, а не этот эффект.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  const speaking = voiceOn
  // Чьи реплики уже звучали: их текст остаётся на экране и после того, как голос замолчал.
  const [heard, setHeard] = useState<ReadonlySet<string>>(new Set())
  useEffect(() => {
    if (voiceOn || voiceLoading) setHeard((before) => (before.has(active.id) ? before : new Set(before).add(active.id)))
  }, [voiceOn, voiceLoading, active.id])

  // Чьи реплики уже появились целиком: при возвращении к эксперту текст стоит на месте, а не появляется заново.
  const [shown, setShown] = useState<ReadonlySet<string>>(new Set())

  const go = (offset: number) => {
    const next = personas[Math.min(personas.length - 1, Math.max(0, index + offset))]
    if (next.id !== active.id) {
      speech.stop()
      onActivate(next.id)
    }
  }

  const drag = useRef<number | null>(null)
  // Перелистывание пальцем заканчивается «нажатием» на аватар под ним — его не считаем.
  const swiped = useRef(false)
  const listen = () => {
    if (voiceOn || voiceLoading) return speech.stop()
    if (!speech.auto) speech.setAuto(true)
    void narrate(key, line, active.id)
  }

  return (
    <section className="gallery" aria-label="Выбор эксперта">
      <div className="gallery-head">
        <h1>Кто ответит на ваш вопрос о сне?</h1>
        <p>Листайте: все читают одни и те же исследования, но объясняют их по-разному.</p>
      </div>

      <div
        className="stage"
        onPointerDown={(event) => {
          drag.current = event.clientX
          swiped.current = false
        }}
        onPointerUp={(event) => {
          const delta = drag.current === null ? 0 : event.clientX - drag.current
          drag.current = null
          if (Math.abs(delta) > 40) {
            swiped.current = true
            go(delta < 0 ? 1 : -1)
          }
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
              data-wait={offset === 0 && waitingTap}
              aria-label={offset !== 0 ? `Выбрать: ${persona.name}` : speaking || voiceLoading ? `${persona.name}: остановить` : heard.has(persona.id) ? `${persona.name}: послушать ещё раз` : `${persona.name}: послушать ответ`}
              onClick={() => {
                if (swiped.current) return
                if (offset !== 0) go(offset)
                else listen()
              }}
            >
              <Avatar
                persona={persona}
                size={208}
                state={offset !== 0 ? "idle" : speaking ? "speaking" : voiceLoading ? "thinking" : "active"}
                getLevel={voiceOn ? speech.level : undefined}
                getMouth={voiceOn ? speech.mouth : undefined}
              />
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

      {speech.error?.key === key && (
        <p className="note" role="alert">
          {speech.error.message}
        </p>
      )}

      <div className="gallery-name" key={`name-${active.id}`}>
        <h2>{active.name}</h2>
        <p>{active.tagline}</p>
      </div>

      {/* Реплика устроена как голосовое сообщение эксперта: слева кнопка «играть», справа — текст: он появляется сразу, голос читает его следом. */}
      <div className="quote">
        <button
          type="button"
          className="quote-play"
          data-wait={waitingTap}
          onClick={listen}
          aria-label={speaking || voiceLoading ? "Остановить" : heard.has(active.id) ? `Послушать ещё раз, как отвечает ${active.name}` : `Послушать, как отвечает ${active.name}`}
        >
          {voiceLoading ? <LoaderCircle size={18} className="spin" /> : speaking ? <Square size={14} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
        </button>
        <div className="quote-body">
          <span className="quote-label">{active.sample ? `На вопрос «${SAMPLE_QUESTION}» отвечает так` : "Как думает"}</span>
          {shown.has(active.id) ? (
            <p className="quote-text">{line}</p>
          ) : heard.has(active.id) ? (
            <div className="quote-text" aria-live="polite">
              <TypedText text={line} onDone={() => setShown((before) => new Set(before).add(active.id))} />
            </div>
          ) : (
            <p className="quote-text quote-hint">Нажмите на кнопку или на самого эксперта — он ответит вслух, а текст появится сразу.</p>
          )}
        </div>
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
