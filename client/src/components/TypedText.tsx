import { useEffect, useMemo, useRef } from "react"

type Props = {
  text: string
  /**
   * Сколько слов уже произнёс голос: текст идёт за ним. Infinity — голос закончил.
   * null — голоса нет, текст печатается сам с ровной скоростью.
   */
  words: number | null
  /** Темп эксперта: больше — медленнее. Влияет только на печать без голоса. */
  pace?: number
  /** Печать без голоса дошла до конца. */
  onDone?: () => void
}

/** Сколько последних букв проявляется одновременно: у текста мягкий край, а не ступенька. */
const EDGE = 9
/** Скорость печати без голоса, букв в секунду. Около 5 мс на букву читается ровно и не кажется медленным. */
const SPEED = 180

/**
 * Текст, который проявляется по буквам, как в привычных ИИ-чатах: ровное движение и мягкий передний край.
 * Место под весь текст занято сразу, поэтому строки не прыгают и то, что ниже, не сдвигается.
 * Буквы красит цикл кадров напрямую, без перерисовки React: на длинном ответе это сотни элементов.
 */
export function TypedText({ text, words, pace = 1, onDone }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const target = useRef(words)
  target.current = words
  const done = useRef(onDone)
  done.current = onDone

  // Абзацы → слова → буквы. У каждого слова запомнено, сколько букв набралось к его концу.
  const { paragraphs, ends, total } = useMemo(() => {
    const ends: number[] = []
    let total = 0
    const paragraphs = text
      .split(/\n{2,}/)
      .map((block) => block.trim())
      .filter(Boolean)
      .map((block) =>
        block.split(/\s+/).map((word) => {
          const letters = Array.from(word)
          total += letters.length
          ends.push(total)
          return letters
        }),
      )
    return { paragraphs, ends, total }
  }, [text])

  useEffect(() => {
    const element = root.current
    if (!element) return
    const letters = element.querySelectorAll<HTMLElement>(".ch")
    const finish = () => {
      element.dataset.done = "true"
      if (target.current === null) done.current?.()
    }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return finish()

    let frame = 0
    let position = 0
    let speed = 0
    let solid = 0
    let last = performance.now()
    const started = last
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const spoken = target.current
      if (spoken === null) {
        // По часам, а не по кадрам: если вкладку свернули, к возвращению текст уже на месте.
        position = ((now - started) / 1000) * (SPEED / pace)
      } else {
        // За голосом: цель — конец слова, которое он сейчас произносит. Скорость тянется к цели плавно,
        // поэтому буквы идут ровно, хотя слова приходят рывками.
        const goal = spoken === Infinity ? total + EDGE : spoken <= 0 ? 0 : (ends[Math.min(ends.length - 1, Math.floor(spoken))] ?? 0)
        const wanted = Math.max(0, (goal - position) * 5 + (goal > position ? 6 : 0))
        speed += (wanted - speed) * (1 - Math.exp(-dt * 12))
        position = Math.min(goal, position + speed * dt)
      }
      const front = Math.min(letters.length, Math.ceil(position))
      for (let index = solid; index < front; index++) {
        const shown = Math.min(1, (position - index) / EDGE)
        letters[index].style.opacity = String(shown * shown * (3 - 2 * shown))
        if (shown >= 1 && index === solid) solid++
      }
      if (solid >= letters.length) return finish()
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [ends, total, pace])

  return (
    <div className="typed" ref={root}>
      {paragraphs.map((block, position) => (
        <p key={position}>
          {block.map((word, inner) => (
            <span key={inner}>
              {word.map((letter, at) => (
                <span key={at} className="ch">
                  {letter}
                </span>
              ))}{" "}
            </span>
          ))}
        </p>
      ))}
    </div>
  )
}
